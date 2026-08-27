# Patterns, term sets and output values the literature pair stage matches on.
from __future__ import annotations

import re

# The two full-corpus screens. Same corpus and config; rep2 differs only by its
# --label, which gave it a distinct run_id, so neither is a retry of the other
# and their pair calls are independent samples.
PRIMARY_RUN_ID = 'screen__v1__cue-abs__847b24'
REPLICATE_RUN_ID = 'screen__v1__cue-abs__rep2__847b24'
RUN_LABELS = {PRIMARY_RUN_ID: 'primary', REPLICATE_RUN_ID: 'rep2'}

# The pipeline stage the pair table publishes under, and its file stem.
PAIRS_STAGE = 'pairs'
PAIRS_NAME = 'literature_pairs'

# The model pick publishes as its own stage rather than as a new version of the
# pair table, so both stay live: the pairs stage is the deterministic assignment
# and this one is that table with the picked accessions folded in, and a consumer
# chooses which mapping risk it accepts.
ACCESSIONS_STAGE = 'accessions'
ACCESSIONS_NAME = 'literature_pairs_picked'

# The agentic curation of the resolved set, which is the deliverable. Its own
# stage again, for the same reason: the table it writes carries a per-pair
# judgement the two mapping stages have no opinion about.
CURATION_STAGE = 'curation'
CURATION_NAME = 'literature_pairs_curated'

# The pair relationship the benchmark treats as a negative. 'both' is dropped:
# it cost 10.8 points of gold recall for 12% of volume.
NO_INTERACTION = 'no_interaction'

# Why a pair was dropped, recorded rather than deleted. An empty drop_reason is
# a kept pair.
DROP_UNGROUNDED = 'ungrounded'
DROP_CONSTRUCT = 'construct'
DROP_REAGENT = 'reagent'
DROP_VIRUS_VIRUS = 'virus_virus'

# The final filter, which runs after curation and decides what ships as a
# Negatome source. These four differ from the four above in what they act on: a
# pair rather than a row, since by this point every row judging a pair has been
# read and the question is what the pair as a whole is worth. There is no
# virus-virus reason here because apply_final_filters already removed those.
NEGATIVES_STEM = 'literature_negatives'
DROP_CONTRADICTED = 'contradicted'
DROP_INTACT = 'intact_positive'
DROP_KNOWN_POSITIVE = 'structural_positive'
DROP_KNOWN_NEGATIVE = 'known_negative'
NEGATIVES_DROP_ORDER = (
    DROP_CONTRADICTED, DROP_INTACT, DROP_KNOWN_POSITIVE, DROP_KNOWN_NEGATIVE,
)

# Name normalisation. Greek letters are spelled out rather than stripped:
# dropping them collapses importin-alpha onto importin-beta.
GREEK = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
    'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
    'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
}
GREEK_TABLE = str.maketrans(GREEK)
NON_ALNUM_RE = re.compile(r'[^a-z0-9]+')

# An EC number names an activity shared by many proteins, never one protein.
EC_NAME_RE = re.compile(r'^EC \d+\.[\d\-]+\.[\d\-]+\.[\d\-]+$')

# Names too generic to identify a protein, before species is even considered.
JUNK_NAMES = frozenset({
    'protein', 'proteins', 'uncharacterized protein', 'putative protein',
    'hypothetical protein', 'deleted', 'fragment', 'peptide', 'enzyme',
})

# Where a paper states what it studied. Results are excluded deliberately: a
# results section names every organism whose protein it compares against.
SCAN_SECTIONS = ('title', 'abstract', 'methods')

# A parenthetical qualifying a strain or isolate rather than naming the organism
# in a form a paper would write.
STRAIN_RE = re.compile(
    r'\b(strain|isolate|serotype|subsp|var)\b', re.IGNORECASE,
)
ORGANISM_PAREN_RE = re.compile(r'\(([^()]+)\)')
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.'-]*")

# Schwartz-Hearst (2003): a candidate abbreviation sits inside parentheses and
# its long form is the text immediately before them.
DEFINITION_PAREN_RE = re.compile(r'\(([^()]{2,30})\)')

# Leading words a definition uses to qualify a protein rather than name it, and
# trailing nouns that generalise it. Stripping both is what turns a long form
# from a phrase into an index key: 'human dipeptidyl peptidase-4' and
# 'N. benthamiana SKP1 ortholog' both fail an exact lookup until reduced.
# Measured to roughly triple the yield.
LEAD_DROP = frozenset({
    'p', 'phospho', 'phosphorylated', 'ac', 'acetyl', 'acetylated', 'ub',
    'ubiquitin', 'ubiquitinated', 'me', 'methyl', 'methylated', 'sumoylated',
    'cleaved', 'active', 'inactive', 'soluble', 'recombinant', 'purified',
    'endogenous', 'exogenous', 'total', 'full', 'fulllength', 'wt', 'wildtype',
    'human', 'mouse', 'rat', 'murine', 'bovine', 'yeast', 'nuclear',
    'cytosolic', 'mature', 'pro', 'pre', 'anti', 'e', 'coli', 'escherichia',
    'saccharomyces', 'cerevisiae', 'drosophila', 'arabidopsis', 'thaliana',
    'oryza', 'sativa', 'the', 'a', 'an', 'its', 'type', 'isoform', 'form',
    'protein', 'putative', 'novel', 'plant', 'bacterial', 'viral', 'fungal',
    'truncated', 'tagged', 'labeled', 'labelled', 'nucleotide', 'free',
    'monomeric',
})
TRAILING_DROP = frozenset({
    'protein', 'proteins', 'peptide', 'polypeptide', 'subunit', 'isoform',
    'chain', 'antigen', 'enzyme', 'gene', 'mrna', 'construct', 'fusion',
    'domain', 'receptor', 'channel', 'kinase', 'complex', 'homolog',
    'homologue', 'orthologue', 'ortholog', 'protease', 'transporter', 'factor',
})
ABBREV_SPECIES_RE = re.compile(r'^[A-Z]\.\s*[a-z]+\s+')

# Single-letter species prefixes and the organism each conventionally denotes.
# c, v, p, s and m also spell cellular, viral, phospho, soluble and
# mitochondrial, so only letters with no competing convention are kept; 'm'
# stays because mouse dominates it, but it is the weakest of the set.
SINGLE_LETTER_ORGANISMS = {
    'h': ('homo sapiens',),
    'm': ('mus musculus',),
    'r': ('rattus norvegicus',),
    'd': ('drosophila', 'dictyostelium'),
    'x': ('xenopus',),
    'z': ('danio rerio',),
    'y': ('saccharomyces cerevisiae',),
    'b': ('bos taurus',),
}
TWO_LETTER_PREFIX_RE = re.compile(r'^([A-Z][a-z])([A-Z0-9].*)$')
ONE_LETTER_PREFIX_RE = re.compile(r'^([a-z])([A-Z].*)$')
YEAST_P_SUFFIX_RE = re.compile(r'^(.*[A-Za-z0-9])p$')

PREFIX_TWO_LETTER = 'two_letter'
PREFIX_ONE_LETTER = 'one_letter'
PREFIX_YEAST_P = 'yeast_p_suffix'

# The light-touch construct filter. It matches only notation that unambiguously
# marks a construct. The broad variant additionally matched variant/del/domain/
# fragment/chimera/fusion/dead/deficient and any single-digit point mutation; it
# removes more but wrongly drops 'DEAD-box protein 5', 'S100B', 'H2A', 'S6K' and
# 'TIR domain-containing protein'. Do not build the broad variant.
DELTA_RE = re.compile(r'[Δ∆]')
MUTANT_WORD_RE = re.compile(
    r'\b(mutant|mutants|mutation|truncation|truncated|deletion|deleted)\b',
    re.IGNORECASE,
)
# A parenthesised residue range, e.g. 'PARP1 (1-214)'.
RESIDUE_RANGE_RE = re.compile(r'\(\s*\d+\s*[-‐-―]\s*\d+\s*\)')
# A stated span, e.g. 'residues 1-214' or '1-214 aa'.
RESIDUE_SPAN_RE = re.compile(
    r'\b(residues?|aa|amino\s+acids?)\b\s*\d+\s*[-‐-―]\s*\d+'
    r'|\b\d+\s*[-‐-―]\s*\d+\s*(residues?|aa|amino\s+acids?)\b',
    re.IGNORECASE,
)
# A point mutation of two or more digits on a stem. Two digits separates
# 'p27T187A' from the proteins 'H2A' and 'E2F1'; the trailing word boundary
# saves 'S100A9'. The stem must end in a digit or lower-case letter, or the
# pattern eats the gene-naming convention outright ('CD40L', 'VPS33B',
# 'BCL11A'). The cost is a mutation on an all-capital stem, so 'BRAFV600E' is
# missed, and the gene-name class is two orders of magnitude the larger.
POINT_MUTATION_RE = re.compile(r'(?<=[0-9a-z])[A-Z]\d{2,}[A-Z]\b')

# Affinity and fluorescent tags, as a leading prefix only. SUMO, TRX and CFP are
# deliberately absent: here they are the protein far more often than the tag, as
# 'SUMO-1' is a sumoylation substrate and 'CFP-10' the Mycobacterium antigen.
TAG_PREFIX_RE = re.compile(
    r'^\s*(GST|MBP|His\d*|\d*x?His|HA|FLAG|Myc|V5|Strep|GFP|YFP|RFP|mCherry)'
    r'\s*[-‐-―_\s]',
    re.IGNORECASE,
)
CONSTRUCT_PATTERNS = (
    DELTA_RE, MUTANT_WORD_RE, RESIDUE_RANGE_RE, RESIDUE_SPAN_RE,
    POINT_MUTATION_RE, TAG_PREFIX_RE,
)

# A token that is nothing but a mutation, which the pattern above cannot see:
# its stem-ending requirement means it never fires across a space or hyphen,
# leaving 'CLINT1 T294A' and 'FUS-P525L' looking like plain names. Only safe
# behind the resolution gate, since alone it eats 'S100B' and the poxvirus ORF
# names 'A46R', 'B18R', 'D10R'.
STANDALONE_MUTATION_RE = re.compile(r'^[A-Z]\d{2,}[A-Z]$')
TOKEN_SPLIT_RE = re.compile(r'[\s‐-―-]+')

# Names that are not a protein under study but that UniProt resolves anyway, so
# mapping cannot remove them. They reach the pair table because a pulldown
# control ('Sro7p did not interact with GST alone') reads to the screen exactly
# like a negative result.
#
# Every entry matches the ENTIRE normalised name and never a substring, or the
# list destroys 'NADP-dependent malic enzyme 4' (gene NADP-ME4) while trying to
# remove a bare NADP. Hand-picked rather than generated: the obvious generator,
# whole-name ChEBI hits among names that resolve, returns 684 names of which
# most are real proteins (Tau, Fas, AP-1, Ras, paxillin, Bim, Met). Adding a
# name is a judgement that it is essentially never a study subject here.
#
# The test for anything added later: block a name whose assigned accession is
# arbitrary (GST lands on a bacterial glutathione S-transferase, LPS on mouse
# TLR4 through the Lps gene synonym), and think again where it is right. Avidin
# and streptavidin are the deliberate exception, mapping correctly but blocked
# because the usage here is overwhelmingly a coated bead.
#
# Deliberately left off, each checked against its excerpts: SUMO, TRX and
# ubiquitin are substrates in their own right; HA is influenza haemagglutinin as
# often as the tag; Myc, Met, Trp, Sec, Ser, Asp and CAT collide with an amino
# acid or reagent; PLP is myelin proteolipid protein, PIP prolactin-induced
# protein, PapC a pilus usher, Ros a receptor kinase, NH4 a rice NPR homologue;
# CSA is cyclosporin A and Cockayne syndrome A in equal measure. Six more came
# off in review because the corpus form is a protein: RFP is Ret finger protein,
# CO2 cytochrome oxidase subunit II, FlaG the flagellar protein, Atp4 yeast ATP
# synthase subunit 4, Kbr Kibra, UreA the urease alpha subunit. HSPG and CSPG
# are protein material with no single accession, a separate open question.
REAGENT_FAMILIES = {
    # Affinity and epitope tags, as a bare side. The leading-tag case
    # ('GST-Sso1p') is the construct filter's job; this is 'GST alone'. A name
    # that says 'tag' is unambiguous whatever the tag is, so 'HA tag' and 'Myc
    # tag' are here while the bare 'HA' and 'Myc' are not.
    'tag': frozenset({
        'gst', 'mbp', 'his', 'his6', '6xhis', 'polyhistidine', 'v5',
        'strep', 'halotag', 'snaptag',
        'histag', 'gsttag', 'mbptag', 'flagtag', 'hatag', 'myctag', 'gfptag',
        'v5tag', 'streptag', 'sumotag', 'epitopetag',
    }),
    # Fluorescent proteins and reporter enzymes, which appear as the negative
    # control arm of a co-IP or two-hybrid assay.
    'reporter': frozenset({
        'gfp', 'egfp', 'yfp', 'eyfp', 'cfp', 'ecfp', 'mrfp', 'mcherry',
        'dsred', 'tdtomato', 'luciferase', 'fireflyluciferase',
        'renillaluciferase', 'lacz', 'betagalactosidase', 'gus',
        'betaglucuronidase',
    }),
    # Blocking agents and pulldown matrix.
    'matrix': frozenset({
        'bsa', 'bovineserumalbumin', 'streptavidin', 'avidin', 'neutravidin',
        'proteina', 'proteing', 'proteinag', 'sepharose', 'agarose',
    }),
    # Nucleotides, cofactors, lipids, detergents and buffer components.
    'chemical': frozenset({
        'atp', 'adp', 'amp', 'camp', 'cgmp', 'gtp', 'gdp', 'utp', 'udp',
        'datp', 'dctp', 'dgtp', 'dttp', 'dump', 'atpgammas',
        'gtpgammas', 'nad', 'nadh', 'nadp', 'nadph', 'fad', 'fmn', 'coa',
        'accoa', 'heme', 'haem', 'pip2', 'pip3', 'thf', 'ipp', 'peg', 'sds',
        'ctab', 'egta', 'edta', 'dtt', 'naf', 'hcl',
        'lps', 'lipopolysaccharide', 'gm1', 'gm2', 'gd1a', 'gd1b', 'hdl',
        'ldl', 'dppc', 'dmpc', 'dlpc', 'pope', 'lpc',
    }),
    # Nucleic acids whose bare name resolves to an accession. The wider
    # nucleic-acid vocabulary belongs to the filter gated on names resolving to
    # nothing, which by construction never sees these.
    'nucleic': frozenset({
        'dna', 'ssdna', 'dsdna', 'rna', 'mrna', 'trna', 'rrna', 'sirna',
        'shrna', 'mirna', 'cdna', 'gdna',
    }),
}
REAGENT_NAMES = frozenset().union(*REAGENT_FAMILIES.values())

# A bare ion, e.g. 'Ca2+', 'Mg2+'. Matched on the written name because
# normalisation strips the charge, and 'Ca2+' then collides with 'CA2',
# carbonic anhydrase 2.
ION_RE = re.compile(r'^[A-Z][a-z]?\d*[+-]+$')

# What resolving one written name against the reviewed index produced.
SIDE_DETERMINED = 'determined_by_name'
SIDE_DETERMINED_BY_TEXT = 'determined_by_text'
SIDE_SPECIES_PICK = 'species_pick'
SIDE_IDENTITY_PICK = 'identity_pick'
SIDE_UNRESOLVED = 'unresolved'
SIDE_TOO_WIDE = 'too_wide'
SIDE_MODEL_PICKED = 'model_picked'

# Which of the paper's own statements supplied a side's organism, strongest
# first: a definition naming an organism for this exact written name beats a
# species prefix on the name, which beats the organism the paper names anywhere.
SOURCE_DEFINITION = 'definition'
SOURCE_PREFIX = 'prefix'
SOURCE_PAPER = 'paper_scan'
SOURCE_NONE = ''

# How a side's accession was arrived at. The only mechanism a consumer has for
# controlling mapping risk now that confidence gating is measured and dead:
# model-picked rows are flagged by construction rather than by a threshold.
PROVENANCE_BY_NAME = 'deterministic_by_name'
PROVENANCE_BY_TEXT = 'deterministic_by_text'
PROVENANCE_MODEL_PICKED = 'model_picked'
PROVENANCE_NONE = ''
OUTCOME_PROVENANCE = {
    SIDE_DETERMINED: PROVENANCE_BY_NAME,
    SIDE_DETERMINED_BY_TEXT: PROVENANCE_BY_TEXT,
    SIDE_MODEL_PICKED: PROVENANCE_MODEL_PICKED,
}

# What a pair still needs, which is whichever of its two sides needs more. A
# model-picked side counts as resolved on a weaker basis than a deterministic one
# - 85-87% precision against 100% by construction - which is what provenance
# records.
STATUS_RESOLVED = 'resolved'
STATUS_NEEDS_MODEL_PICK = 'needs_model_pick'
STATUS_NEEDS_CURATION = 'needs_curation'

# Weakest last. assign_status takes the later of a pair's two sides.
STATUS_ORDER = (
    STATUS_RESOLVED, STATUS_NEEDS_MODEL_PICK, STATUS_NEEDS_CURATION,
)
SIDE_STATUS = {
    SIDE_DETERMINED: STATUS_RESOLVED,
    SIDE_DETERMINED_BY_TEXT: STATUS_RESOLVED,
    SIDE_MODEL_PICKED: STATUS_RESOLVED,
    SIDE_SPECIES_PICK: STATUS_NEEDS_MODEL_PICK,
    SIDE_IDENTITY_PICK: STATUS_NEEDS_MODEL_PICK,
    SIDE_UNRESOLVED: STATUS_NEEDS_CURATION,
    SIDE_TOO_WIDE: STATUS_NEEDS_CURATION,
}

# The side outcomes the model pick answers. Kept beside SIDE_STATUS: an outcome
# added there as STATUS_NEEDS_MODEL_PICK but not here would be routed to the pick
# and then never asked about.
PICK_OUTCOMES = (SIDE_SPECIES_PICK, SIDE_IDENTITY_PICK)

# Which text a side's pick was made on. The paragraph its excerpt was copied from
# where one was found, and the excerpt alone where none was - the weaker arm, at
# 64.1% accuracy against the paragraph's 75.6%.
PARAGRAPH_FROM_BLOCK = 'block'
PARAGRAPH_FROM_EXCERPT = 'excerpt'

# What the model pick did with one side. Only PICK_ANSWERED assigns an accession;
# the other three leave the pair needing a pick and are kept apart because each
# says something different - an abstention is the model doing what it was told, a
# no-answer is retryable, and an ungrounded pick is one whose quoted evidence is
# not a verbatim span of the paragraph it was shown, so it is thrown away.
PICK_NONE = ''
PICK_ANSWERED = 'answered'
PICK_ABSTAINED = 'abstained'
PICK_UNGROUNDED = 'ungrounded'
PICK_NO_ANSWER = 'no_answer'
PICK_LABELS = {
    PICK_ANSWERED: 'answered', PICK_ABSTAINED: 'abstained',
    PICK_UNGROUNDED: 'ungrounded evidence, pick voided',
    PICK_NO_ANSWER: 'no answer from the API',
}
PICK_ORDER = (PICK_ANSWERED, PICK_ABSTAINED, PICK_UNGROUNDED, PICK_NO_ANSWER)

# What curation did with one pair. The failures are kept apart because they call
# for different things: an ungrounded verdict is thrown away, a no-answer is
# retryable, and a pair the model never judged says the packet and the answer
# disagree about what was asked.
CURATION_NONE = ''
CURATION_JUDGED = 'judged'
CURATION_UNGROUNDED = 'ungrounded'
CURATION_NO_ANSWER = 'no_answer'
CURATION_NOT_JUDGED = 'not_judged'
CURATION_SELF_PAIR = 'self_pair'
CURATION_LABELS = {
    CURATION_JUDGED: 'judged',
    CURATION_UNGROUNDED: 'ungrounded evidence, verdict voided',
    CURATION_NO_ANSWER: 'no answer from the API',
    CURATION_NOT_JUDGED: 'in the packet, absent from the answer',
    CURATION_SELF_PAIR: 'one accession on both sides, never sent',
}
CURATION_ORDER = (
    CURATION_JUDGED, CURATION_UNGROUNDED, CURATION_NO_ANSWER,
    CURATION_NOT_JUDGED, CURATION_SELF_PAIR,
)

# The two relationships that assert there is nothing to quote, so an empty
# evidence sentence under either is consistent rather than ungrounded.
NO_CLAIM_RELATIONSHIPS = ('not_discussed', 'no_evidence')

# The three conditions a pair must meet to enter the benchmark as a negative. The
# model reports usable_as_negative itself; apply recomputes it from these so a
# verdict that says true while its own fields say otherwise cannot publish a pair.
# The relationship value is the rubric's spelling and is deliberately not
# NO_INTERACTION above, which is the screen's: the two stages answer different
# questions.
USABLE_MAPPING = 'correct'
USABLE_RELATIONSHIP = 'non_interaction'
USABLE_QUALIFIER = 'full_length_pair'

# Stated once, so the gate applied to a single verdict mid-run and the gate
# applied to the whole table at publish cannot disagree. Grounding is checked
# alongside these and is not a field comparison, so it is not in here.
USABLE_CONDITIONS = (
    ('mapping_verdict', USABLE_MAPPING),
    ('relationship', USABLE_RELATIONSHIP),
    ('qualifier', USABLE_QUALIFIER),
)
