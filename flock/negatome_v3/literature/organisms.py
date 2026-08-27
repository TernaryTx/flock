# Species evidence a paper carries about the proteins it names.
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pyarrow.parquet as pq

from flock.negatome_v3.literature import CORPUS_BLOCKS_PATH
from flock.negatome_v3.literature import PAIRS_ROOT
from flock.negatome_v3.literature import PAPER_DEFINITIONS_PATH
from flock.negatome_v3.literature import PAPER_ORGANISMS_PATH
from flock.negatome_v3.literature.names import get_name_index
from flock.negatome_v3.literature.names import normalise_name
from flock.negatome_v3.literature.names import UniprotNameIndex
from flock.negatome_v3.literature.vocabulary import ABBREV_SPECIES_RE
from flock.negatome_v3.literature.vocabulary import DEFINITION_PAREN_RE
from flock.negatome_v3.literature.vocabulary import LEAD_DROP
from flock.negatome_v3.literature.vocabulary import ONE_LETTER_PREFIX_RE
from flock.negatome_v3.literature.vocabulary import ORGANISM_PAREN_RE
from flock.negatome_v3.literature.vocabulary import PREFIX_ONE_LETTER
from flock.negatome_v3.literature.vocabulary import PREFIX_TWO_LETTER
from flock.negatome_v3.literature.vocabulary import PREFIX_YEAST_P
from flock.negatome_v3.literature.vocabulary import SCAN_SECTIONS
from flock.negatome_v3.literature.vocabulary import SINGLE_LETTER_ORGANISMS
from flock.negatome_v3.literature.vocabulary import STRAIN_RE
from flock.negatome_v3.literature.vocabulary import TOKEN_RE
from flock.negatome_v3.literature.vocabulary import TRAILING_DROP
from flock.negatome_v3.literature.vocabulary import TWO_LETTER_PREFIX_RE
from flock.negatome_v3.literature.vocabulary import YEAST_P_SUFFIX_RE

logger = logging.getLogger(__name__)


class PaperEvidence(NamedTuple):
    """What one paper's own text says about organisms and its abbreviations."""

    organism_ids: frozenset[int]
    definitions: dict[str, list[str]]


class SpeciesPrefix(NamedTuple):
    """A species prefix read off a written name.

    Args:
        rule: Which of the prefix rules fired.
        stem: The name with the prefix removed, which is what the index is
            looked up on when the written name resolves to nothing.
        taxids: The organisms the prefix can denote.
    """

    rule: str
    stem: str
    taxids: frozenset[int]


@lru_cache(maxsize=None)
def split_organism(organism: str) -> tuple[str, tuple[str, ...]]:
    """Split a UniProt organism string into its scientific and common names.

    UniProt writes 'Homo sapiens (Human)' and 'Canis lupus familiaris (Dog)
    (Canis familiaris)', so the parentheticals supply common names for free
    rather than needing a hand-written list. Strain and isolate qualifiers are
    dropped: they name a laboratory stock, not something a paper writes.

    Cached because the callers walk all 575k reviewed accessions and only 15k
    distinct organism strings exist among them, several times over.

    Args:
        organism: The organism field as UniProt writes it.

    Returns:
        The scientific name, and each usable common name.
    """
    scientific = organism.split(' (')[0].strip()
    commons = [
        item.strip() for item in ORGANISM_PAREN_RE.findall(organism)
        if item.strip() and not STRAIN_RE.search(item)
    ]
    return scientific, tuple(
        name for name in commons if 1 <= len(name.split()) <= 2
    )


def build_organism_terms(
        index: UniprotNameIndex,
        taxids: set[int] | None = None,
) -> dict[str, frozenset[int]]:
    """Build the term dictionary the text scan looks organisms up in.

    Each organism contributes its two-word scientific name, the abbreviated
    'G. species' form a paper is more likely to write, and any common name
    UniProt records for it.

    A term dictionary rather than one regex per organism keeps the scan linear
    in the length of the text instead of linear in the number of organisms.

    Args:
        index: The reviewed-UniProt name index.
        taxids: Restrict to these organisms. Passing the taxids actually
            reachable from the pair set keeps rare organisms from claiming a
            common word.

    Returns:
        Lower-cased term to the taxids it can denote. A term is one or two
        words, matching how the scan looks up n-grams.
    """
    terms: dict[str, set[int]] = defaultdict(set)
    for card in index.cards.values():
        if taxids is not None and card.organism_id not in taxids:
            continue
        scientific, commons = split_organism(card.organism)
        words = scientific.split()
        if len(words) >= 2 and words[0][:1].isalpha():
            terms[f'{words[0]} {words[1]}'.casefold()].add(card.organism_id)
            terms[f'{words[0][0]}. {words[1]}'.casefold()].add(
                card.organism_id,
            )
        for common in commons:
            terms[common.casefold()].add(card.organism_id)
    return {term: frozenset(found) for term, found in terms.items()}


def scan_organisms(text: str, terms: dict[str, frozenset[int]]) -> set[int]:
    """Return every taxid the text names, by one- and two-gram lookup.

    Args:
        text: The paper text to scan.
        terms: The term dictionary from build_organism_terms.

    Returns:
        The taxids named. Empty if the text names none.
    """
    tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    found: set[int] = set()
    for position, token in enumerate(tokens):
        single = terms.get(token)
        if single:
            found |= single
        if position + 1 < len(tokens):
            pair = terms.get(f'{token} {tokens[position + 1]}')
            if pair:
                found |= pair
    return found


def valid_short_form(candidate: str) -> bool:
    """Return whether parenthesised text can be an abbreviation.

    Args:
        candidate: The text inside the parentheses.

    Returns:
        Whether it satisfies the Schwartz-Hearst conditions on a short form.
    """
    candidate = candidate.strip()
    if not 2 <= len(candidate) <= 10:
        return False
    if len(candidate.split()) > 2:
        return False
    if not candidate[0].isalnum():
        return False
    return any(char.isalpha() for char in candidate)


def find_long_form(window: str, short: str) -> str | None:
    """Match an abbreviation back into the text preceding it, right to left.

    Every character of the short form must appear in the window in order, read
    right to left, and the short form's first character must align with the
    start of a word.

    Args:
        window: Text immediately before the open parenthesis.
        short: The abbreviation.

    Returns:
        The matched long form, or None if the window does not support one.
    """
    short_index = len(short) - 1
    window_index = len(window) - 1
    while short_index >= 0:
        char = short[short_index].casefold()
        if not char.isalnum():
            short_index -= 1
            continue
        while window_index >= 0:
            if window[window_index].casefold() == char and (
                short_index > 0 or _starts_word(window, window_index)
            ):
                break
            window_index -= 1
        if window_index < 0:
            return None
        if short_index == 0:
            break
        short_index -= 1
        window_index -= 1
    if window_index < 0:
        return None
    long_form = window[window_index:].strip()
    words = long_form.split()
    letters = sum(1 for char in short if char.isalnum())
    if not words or len(words) > min(letters + 5, letters * 2):
        return None
    # A stray opening parenthesis earlier in the sentence leaves the match
    # straddling a boundary it should not cross, which produces long forms like
    # 'M242) or 53 to 86'. An unbalanced closer is the tell.
    if long_form.count(')') > long_form.count('('):
        return None
    return long_form


def _starts_word(text: str, position: int) -> bool:
    """Return whether a position begins a word."""
    return position == 0 or not text[position - 1].isalnum()


def extract_definitions(
        text: str,
        wanted: set[str] | None = None,
        window_chars: int = 200,
) -> list[tuple[str, str]]:
    """Return every (abbreviation, long form) definition a block states.

    Args:
        text: One block of paper text.
        wanted: Normalised abbreviations to keep. Passing the ones the caller
            will actually use skips the right-to-left match on the rest, which
            is most of them: a paper's parentheses are overwhelmingly '(Fig. 2)'
            and '(P < 0.05)', and matching those back into the text is the
            expensive part.
        window_chars: How much text before the parenthesis a long form may span.

    Returns:
        The definitions found, in the order they appear.
    """
    found: list[tuple[str, str]] = []
    for match in DEFINITION_PAREN_RE.finditer(text):
        short = match.group(1).strip()
        if not valid_short_form(short):
            continue
        if wanted is not None and normalise_name(short) not in wanted:
            continue
        window_start = max(0, match.start() - window_chars)
        window = text[window_start:match.start()]
        long_form = find_long_form(window, short)
        if long_form and long_form.casefold() != short.casefold():
            found.append((short, long_form))
    return found


def long_form_variants(long_form: str) -> list[str]:
    """Return readings of a definition phrase a name index might carry.

    A long form is a phrase, not an index key. Leading qualifier and species
    words come off the front, generic nouns off the back, and each intermediate
    reading is kept because the index may carry any one of them.

    Args:
        long_form: The definition as the paper writes it.

    Returns:
        The readings to try, longest first, duplicates removed.
    """
    variants = [long_form]
    stripped = ABBREV_SPECIES_RE.sub('', long_form).strip()
    if stripped != long_form:
        variants.append(stripped)
    words = [word for word in stripped.split() if word]
    start = 0
    while start < len(words) - 1 and normalise_name(words[start]) in LEAD_DROP:
        start += 1
        variants.append(' '.join(words[start:]))
    end = len(words)
    while end > start + 1 and normalise_name(words[end - 1]) in TRAILING_DROP:
        end -= 1
        variants.append(' '.join(words[start:end]))
    return [item for item in dict.fromkeys(variants) if item.strip()]


def scan_corpus(
        blocks_path: Path,
        wanted: dict[str, set[str]],
        terms: dict[str, frozenset[int]],
        sections: tuple[str, ...] = SCAN_SECTIONS,
        char_cap: int = 8000,
) -> dict[str, PaperEvidence]:
    """Collect organism mentions and abbreviation definitions in one pass.

    Both are gathered together because each costs a pass over a 2.83 GiB block
    table, and the table is streamed by row group rather than loaded.

    The two scans read different text. Organisms are read from the title,
    abstract and methods only, capped at char_cap, because those state what the
    paper studied. Definitions are read from every block, because a paper
    introduces an abbreviation wherever it first uses it.

    Args:
        blocks_path: The corpus block table.
        wanted: paper_id to the normalised names that paper needs defined.
        terms: The organism term dictionary.
        sections: Canonical sections the organism scan reads.
        char_cap: Most characters of those sections to scan per paper.

    Returns:
        paper_id to its evidence, for every wanted paper the scan saw.
    """
    scanned: dict[str, list[str]] = defaultdict(list)
    scanned_chars: dict[str, int] = defaultdict(int)
    definitions: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list),
    )
    handle = pq.ParquetFile(blocks_path)
    n_groups = handle.metadata.num_row_groups
    for group in range(n_groups):
        table = handle.read_row_group(
            group, columns=['paper_id', 'canonical_section', 'text'],
        )
        for paper_id, section, text in zip(
            table.column('paper_id').to_pylist(),
            table.column('canonical_section').to_pylist(),
            table.column('text').to_pylist(),
        ):
            needed = wanted.get(paper_id)
            if needed is None or not text:
                continue
            if section in sections and scanned_chars[paper_id] < char_cap:
                room = char_cap - scanned_chars[paper_id]
                scanned[paper_id].append(text[:room])
                scanned_chars[paper_id] += min(len(text), room)
            if needed and '(' in text:
                for short, long_form in extract_definitions(text, needed):
                    key = normalise_name(short)
                    if long_form not in definitions[paper_id][key]:
                        definitions[paper_id][key].append(long_form)
        if group % 20 == 0:
            logger.info(
                'row group %d/%d, %d papers scanned, %d with a definition',
                group, n_groups, len(scanned), len(definitions),
            )
    return {
        paper_id: PaperEvidence(
            organism_ids=frozenset(
                scan_organisms(' '.join(scanned.get(paper_id, [])), terms),
            ),
            definitions=dict(definitions.get(paper_id, {})),
        )
        for paper_id in wanted
        if paper_id in scanned or paper_id in definitions
    }


def build_prefix_taxa(
        index: UniprotNameIndex,
) -> tuple[dict[str, frozenset[int]], dict[int, str]]:
    """Map a two-letter genus-and-species code to the taxids it can denote.

    Built from the reviewed organism strings, so a code whose organism has no
    reviewed entry is simply absent. That is the gap a definition reading
    'P. patens DMC1' fills.

    Args:
        index: The reviewed-UniProt name index.

    Returns:
        Lower-cased two-letter code to taxids, and taxid to organism name.
    """
    taxon_name: dict[int, str] = {}
    codes: dict[str, set[int]] = defaultdict(set)
    for card in index.cards.values():
        if card.organism_id in taxon_name:
            continue
        taxon_name[card.organism_id] = card.organism
        words = split_organism(card.organism)[0].split()
        if len(words) >= 2 and len(words[0]) > 1 and words[1][:1].isalpha():
            codes[f'{words[0][0]}{words[1][0]}'.casefold()].add(
                card.organism_id,
            )
    return {code: frozenset(found) for code, found in codes.items()}, taxon_name


def single_letter_taxa(letter: str, taxon_name: dict[int, str]) -> frozenset[int]:
    """Return the taxids a single-letter species prefix can stand for.

    Args:
        letter: The lower-cased prefix letter.
        taxon_name: taxid to organism name, from build_prefix_taxa.

    Returns:
        The taxids, empty if the letter carries no convention.
    """
    wanted = SINGLE_LETTER_ORGANISMS.get(letter, ())
    if not wanted:
        return frozenset()
    return frozenset(
        taxid for taxid, organism in taxon_name.items()
        if any(organism.casefold().startswith(word) for word in wanted)
    )


def species_prefix(
        name: str,
        codes: dict[str, frozenset[int]],
        single: dict[str, frozenset[int]],
) -> SpeciesPrefix | None:
    """Read a species prefix off a written name.

    Rules are tried longest-first, so 'AtCGL160' is read as the Arabidopsis code
    rather than a single 'a'. The remainder must be at least three characters,
    or a two-character protein name would be split into a prefix and nothing.

    Args:
        name: A protein name as the screen emitted it.
        codes: Two-letter code to taxids, from build_prefix_taxa.
        single: Single letter to taxids, from single_letter_taxa.

    Returns:
        The rule that fired, the name with the prefix removed, and the taxids
        the prefix denotes. None if no rule applies.
    """
    stem = name.strip()
    yeast = YEAST_P_SUFFIX_RE.match(stem)
    if yeast and len(yeast.group(1)) >= 3 and single.get('y'):
        return SpeciesPrefix(PREFIX_YEAST_P, yeast.group(1), single['y'])
    two = TWO_LETTER_PREFIX_RE.match(stem)
    if two and len(two.group(2)) >= 3:
        found = codes.get(two.group(1).casefold())
        if found:
            return SpeciesPrefix(PREFIX_TWO_LETTER, two.group(2), found)
    one = ONE_LETTER_PREFIX_RE.match(stem)
    if one and len(one.group(2)) >= 3:
        found = single.get(one.group(1))
        if found:
            return SpeciesPrefix(PREFIX_ONE_LETTER, one.group(2), found)
    return None


def scan_wanted_papers(
        frame: pd.DataFrame,
        index: UniprotNameIndex,
) -> tuple[dict[str, set[str]], set[int]]:
    """Work out which papers need a scan, and which organisms are reachable.

    Every side that is not already determined by its name alone needs the scan,
    not only the species-ambiguous ones. Widening it this way is what took
    determined-by-text sides from 17,115 to 29,290 when it was measured. The
    test is read off the index rather than off the pair table's outcome column: a
    name carrying exactly one candidate is what the resolve stage calls
    determined_by_name, and asking the index directly keeps this module from
    depending on that stage's vocabulary.

    Args:
        frame: A pair table carrying paper_id, name_a and name_b.
        index: The reviewed-UniProt name index.

    Returns:
        paper_id to the normalised names that paper needs defined, and the
        taxids reachable from any candidate of any name in the table.
    """
    wanted: dict[str, set[str]] = defaultdict(set)
    reachable: set[int] = set()
    seen: dict[str, int] = {}
    for paper_id, name_a, name_b in zip(
        frame['paper_id'], frame['name_a'], frame['name_b'],
    ):
        for name in (str(name_a), str(name_b)):
            n_candidates = seen.get(name)
            if n_candidates is None:
                found = index.candidates(name)
                n_candidates = seen[name] = len(found)
                reachable.update(
                    index.cards[accession].organism_id for accession in found
                )
            if n_candidates != 1:
                wanted[str(paper_id)].add(normalise_name(name))
    return dict(wanted), reachable


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Scan the corpus for the species evidence each paper carries: the '
            'organisms it names, and the abbreviations it defines.'
        ),
    )
    parser.add_argument(
        '--source', default=str(PAIRS_ROOT / 'screen_pairs_resolved.parquet'),
        help='The resolved pair table, which says which papers need a scan.',
    )
    parser.add_argument(
        '--blocks',
        default=str(CORPUS_BLOCKS_PATH),
        help='The corpus block table. Streamed by row group, never loaded.',
    )
    parser.add_argument(
        '--organisms-out', default=str(PAPER_ORGANISMS_PATH),
        help='Where to write one row per paper with the taxids it names.',
    )
    parser.add_argument(
        '--definitions-out', default=str(PAPER_DEFINITIONS_PATH),
        help='Where to write one row per abbreviation a paper defines.',
    )
    return parser.parse_args()


def main() -> None:
    """Scan the corpus for the species evidence each paper carries.

    Two things come out of one streaming pass over the block table, because each
    would otherwise cost its own pass over 2.83 GiB. The first is the set of
    organisms a paper names in its title, abstract and methods, which is the
    evidence the species check uses when a name resolves to one orthologue group
    across several species. The second is the abbreviations the paper defines,
    which is stronger evidence still: a paper writing 'P. patens DMC1 (PpDMC1)'
    ties an organism to that exact written name rather than to the paper as a
    whole.

    Both outputs are keyed on paper_id and are written separately, so the
    wiring step can join either without re-reading the corpus.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args = parse_args()

    frame = pd.read_parquet(args.source)
    index = get_name_index()

    wanted, reachable = scan_wanted_papers(frame, index)
    logger.info('%d papers need a scan', len(wanted))
    terms = build_organism_terms(index, reachable)
    logger.info(
        '%d organism terms over %d reachable organisms',
        len(terms), len(reachable),
    )

    evidence = scan_corpus(Path(args.blocks), wanted, terms)

    organisms = pd.DataFrame(
        [
            {'paper_id': paper_id, 'organism_ids': sorted(item.organism_ids)}
            for paper_id, item in evidence.items()
        ],
    )
    definition_rows = [
        {'paper_id': paper_id, 'abbrev': abbrev, 'long_form': long_form}
        for paper_id, item in evidence.items()
        for abbrev, long_forms in item.definitions.items()
        for long_form in long_forms
    ]
    for path_text, table in (
        (args.organisms_out, organisms),
        (args.definitions_out, pd.DataFrame(definition_rows)),
    ):
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        print(f'wrote {len(table):,} rows to {path}')

    named = organisms['organism_ids'].map(len) if len(
        organisms,
    ) else pd.Series(dtype=int)
    print(f'\npapers scanned: {len(organisms):,}')
    print(f'  naming at least one organism: {(named > 0).sum():,}')
    print(f'  naming exactly one:           {(named == 1).sum():,}')
    print(
        f'  median organisms named:       {named.median() if len(named) else 0:.0f}',
    )
    print(f'papers defining a wanted abbreviation: '
          f'{len({row["paper_id"] for row in definition_rows}):,}')


if __name__ == '__main__':
    main()
