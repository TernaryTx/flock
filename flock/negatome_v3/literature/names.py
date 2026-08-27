# Every name a reviewed UniProt entry goes by, indexed for lookup by written name.
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import pandas as pd

from flock.negatome_v3.literature.vocabulary import EC_NAME_RE
from flock.negatome_v3.literature.vocabulary import GREEK_TABLE
from flock.negatome_v3.literature.vocabulary import JUNK_NAMES
from flock.negatome_v3.literature.vocabulary import NON_ALNUM_RE
from flock.uniprot_metadata import get_reviewed_data
from flock.uniprot_metadata import UniprotReviewedData

logger = logging.getLogger(__name__)


class AccessionCard(NamedTuple):
    """One reviewed UniProt entry, reduced to what name resolution needs.

    A NamedTuple rather than a dataclass because there is one per reviewed
    accession, and 575k dataclass instances cost several times the memory.
    """

    accession: str
    entry_name: str
    stem: str
    protein_name: str
    gene_names: str
    organism: str
    organism_id: int
    is_viral: bool


def normalise_name(name: str) -> str:
    """Canonicalise a protein name for matching.

    Args:
        name: A protein name, as a paper or UniProt writes it.

    Returns:
        The normalised key, empty if nothing survives.
    """
    lowered = name.strip().casefold().translate(GREEK_TABLE)
    return NON_ALNUM_RE.sub('', lowered)


def surface_forms(name: str) -> set[str]:
    """Return the normalised keys a written name may legitimately match on.

    A slash joins two names as often as it spells one, so both readings are
    returned rather than guessing which was meant.

    Args:
        name: A protein name as a paper writes it.

    Returns:
        The normalised keys it may match on, empty strings removed.
    """
    forms = {normalise_name(name)}
    if '/' in name:
        forms |= {normalise_name(part) for part in name.split('/')}
    return {form for form in forms if form}


def split_protein_names(text: str) -> list[str]:
    """Split a UniProt protein-names field into its main and alternative names.

    Parenthesis depth is tracked rather than regex-matched because nested
    parentheses are common in this field and a non-greedy regex truncates them.

    Args:
        text: The 'protein_names' field value.

    Returns:
        The main name followed by each alternative, EC numbers removed.
    """
    main: list[str] = []
    current: list[str] = []
    alternatives: list[str] = []
    depth = 0
    for char in text:
        if char == '(':
            depth += 1
            if depth == 1:
                continue
        elif char == ')':
            depth -= 1
            if depth == 0:
                alternatives.append(''.join(current).strip())
                current = []
                continue
        (main if depth == 0 else current).append(char)
    names = [''.join(main).strip()] + alternatives
    return [name for name in names if name and not EC_NAME_RE.match(name)]


def clean_field(value: object) -> str:
    """Return a stripped string for a field pandas may have read as NaN.

    Args:
        value: The raw field value.

    Returns:
        The stripped text, empty where the field is missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    return str(value).strip()


@dataclass(frozen=True)
class UniprotNameIndex:
    """Every name a reviewed UniProt entry goes by, plus a card per accession.

    Args:
        names: Normalised name to the accessions it can mean.
        cards: Accession to its card.
        release: The UniProt release this was built from, e.g. '2026_02'. Record
            it in any output's provenance: entries are added, renamed and
            demerged between releases, so it is what a rebuild has to match.
    """

    names: dict[str, tuple[str, ...]]
    cards: dict[str, AccessionCard]
    release: str

    def records_name(self, name: str) -> bool:
        """Return whether UniProt records this exact name for a reviewed entry.

        Whole-name only, deliberately not through surface_forms: splitting a
        slashed name would let 'ARF1/Q71L' answer True on its first half.

        Args:
            name: A protein name as a paper writes it.

        Returns:
            True if the normalised name is one UniProt records.
        """
        return normalise_name(name) in self.names

    def candidates(self, name: str) -> tuple[str, ...]:
        """Return every reviewed accession a written name can mean.

        Args:
            name: A protein name as a paper writes it.

        Returns:
            The candidate accessions, sorted so the order is stable.
        """
        found: set[str] = set()
        for form in surface_forms(name):
            found.update(self.names.get(form, ()))
        return tuple(sorted(found))

    def group_candidates(
        self,
        accessions: Iterable[str],
    ) -> list[frozenset[str]]:
        """Partition candidate accessions into orthologue groups by entry-name stem.

        The stem is UniProt naming convention rather than curated orthology:
        TP53_HUMAN and TP53_MOUSE both give TP53. eggNOG and OrthoDB were both
        measured against it and neither is used. OrthoDB cannot be compared by
        string equality at all, since an id carries the clade the group is
        defined at and UniProt cross-references human SUMO1 at Eukaryota against
        mouse SUMO1 at Rodentia. eggNOG groups a textbook orthologue pair
        correctly but over-merges paralogue families, and 'P-glycoprotein'
        collapses six stem groups under it. Under-grouping costs bucket-1 yield;
        over-grouping assigns a confidently wrong accession, so the stem errs the
        safe way.

        Grouping runs over the given accessions only, never the whole index: the
        question is how many distinct proteins are among these candidates, and a
        global partition would be both wrong and far more expensive. An accession
        carrying no stem forms its own group.

        Args:
            accessions: Candidate accessions for one written name. Accessions
                absent from the index are dropped rather than grouped blindly.

        Returns:
            The groups, each a frozenset of accessions, ordered by their
            smallest accession so the result is stable.
        """
        by_stem: dict[str, set[str]] = defaultdict(set)
        stemless: list[frozenset[str]] = []
        for accession in dict.fromkeys(accessions):
            card = self.cards.get(accession)
            if card is None:
                continue
            if card.stem:
                by_stem[card.stem].add(accession)
            else:
                stemless.append(frozenset({accession}))
        return sorted(
            [frozenset(group) for group in by_stem.values()] + stemless,
            key=min,
        )


def build_name_index(
    data: UniprotReviewedData | None = None,
    min_name_chars: int = 2,
) -> UniprotNameIndex:
    """Index every name UniProt records for a reviewed entry.

    Gene symbols and their synonyms, the main protein name and every
    parenthetical alternative, and the entry-name stem are all indexed. Indexing
    all of them is what takes per-side resolution on the literature Negatome from
    27.4% (human reference proteome, primary names only) to 54.5%.

    Because the index is built from the current reviewed release, an accession
    UniProt has deleted or demerged is simply absent: a dead-accession check
    belongs wherever legacy accessions enter from outside, not here.

    Args:
        data: The reviewed table. Defaults to the process-wide shared instance.
        min_name_chars: Shortest normalised name to index. A single character is
            attested in every paper of any length, so it is not evidence.

    Returns:
        The name index, carrying the UniProt release it was built from.
    """
    data = data or get_reviewed_data()
    frame = data.data
    viral = data.get_viral_accessions()

    index: dict[str, set[str]] = defaultdict(set)
    cards: dict[str, AccessionCard] = {}
    n_skipped = 0
    rows = zip(
        frame['accession'], frame['entry_name'], frame['protein_names'],
        frame['gene_names'], frame['organism'], frame['organism_id'],
    )
    for accession, entry_name, protein_names, gene_names, organism, organism_id in rows:
        try:
            taxid = int(organism_id)
        except (TypeError, ValueError):
            n_skipped += 1
            continue
        every_name = split_protein_names(clean_field(protein_names))
        genes = clean_field(gene_names)
        stem = clean_field(entry_name).split('_')[0]
        cards[accession] = AccessionCard(
            accession=accession,
            entry_name=clean_field(entry_name),
            stem=stem,
            protein_name=every_name[0] if every_name else '',
            gene_names=genes,
            organism=clean_field(organism),
            organism_id=taxid,
            is_viral=accession in viral,
        )
        for name in [*every_name, *genes.split(), stem]:
            if name.casefold() in JUNK_NAMES:
                continue
            key = normalise_name(name)
            if len(key) < min_name_chars or key.isdigit():
                continue
            index[key].add(accession)

    if n_skipped:
        logger.warning(
            'Skipped %d reviewed entries with no usable organism id', n_skipped,
        )
    # Frozen to tuples rather than left as sets: at this scale the sets cost
    # several times what the tuples do, and nothing downstream mutates them.
    names = {
        key: tuple(sorted(accessions)) for key, accessions in index.items()
    }
    release = data.get_release()
    logger.info(
        '%d names -> %d accessions, %d entry-name stems, %d organisms, UniProt release %s',
        len(names), len(cards), len({card.stem for card in cards.values()}),
        len({card.organism_id for card in cards.values()}), release,
    )
    return UniprotNameIndex(names=names, cards=cards, release=release)


@lru_cache(maxsize=1)
def get_name_index() -> UniprotNameIndex:
    """Return a process-wide shared name index.

    Building it reads the whole reviewed table and holds several hundred MB, so
    callers should share one instance through this accessor rather than calling
    build_name_index themselves.

    Returns:
        The cached index.
    """
    return build_name_index()
