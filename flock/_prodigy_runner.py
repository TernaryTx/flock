# Batch PRODIGY-cryst runner, executed inside the legacy `prodigy` conda env.
#
# prodigy-cryst 1.0.1 pins scikit-learn==0.22 and numpy<1.21, which cannot coexist
# with the `flock` env's numpy 2.x, so it lives in its own environment and is driven
# as a subprocess. This script is the other side of that boundary: it must import
# nothing beyond the standard library and prodigy_cryst itself, and must stay
# Python 3.8 compatible.
#
# Reads a file of interface PDB paths (one per line), writes TSV to stdout:
#   path <TAB> predicted_class <TAB> prob_bio <TAB> prob_xtal <TAB> n_contacts
#        <TAB> link_density <TAB> error
# A structure that cannot be classified yields an ERROR row rather than aborting
# the batch, so one bad file does not cost the whole run.
from __future__ import annotations

import sys
import warnings

from prodigy_cryst.interface_classifier import ProdigyCrystal
from prodigy_cryst.lib.parsers import parse_structure

# PPI3D writes both partners of an interface as chains A and B in its interface
# coordinate files, whatever the chains were called in the parent entry.
SELECTION = ['A', 'B']


def main():
    warnings.simplefilter('ignore')
    with open(sys.argv[1]) as handle:
        paths = [line.strip() for line in handle if line.strip()]

    for path in paths:
        try:
            structure, _, _ = parse_structure(path)
            prodigy = ProdigyCrystal(structure, SELECTION)
            prodigy.predict()
            predicted_class, prob_bio, prob_xtal = prodigy.predicted_class
            print(
                '%s\t%s\t%.4f\t%.4f\t%d\t%.6f\t' % (
                    path, predicted_class, prob_bio, prob_xtal,
                    len(prodigy.ic_network), prodigy.link_density,
                ),
            )
        except Exception as exc:
            # Five tabs: the four numeric columns are left empty and the reason
            # goes in its own trailing field. Field count must match
            # RUNNER_COLUMNS exactly or classify_interfaces drops the row, so a
            # miscount here loses failures silently rather than reporting them.
            print(
                '{}\tERROR\t\t\t\t\t{}'.format(
                    path, str(exc).replace('\t', ' ').replace('\n', ' '),
                ),
            )
        sys.stdout.flush()


if __name__ == '__main__':
    main()
