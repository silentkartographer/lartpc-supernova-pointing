# Authored by Hilary Utaegbulam 

"""Constants and class mappings."""
from __future__ import annotations
import argparse

Config = argparse.Namespace


# PDG codes

PDG_ELECTRON_POS = 11
PDG_ELECTRON_NEG = -11
PDG_MUON_POS = 13
PDG_MUON_NEG = -13
PDG_MICHEL = 9900011
PDG_BREMS  = 9910000



# Segmentation class IDs

CLASS_BG = 0
CLASS_ELECTRON = 1
CLASS_MUON = 2
CLASS_MICHEL = 3
CLASS_BREMS = 4

CLASS_ID_TO_NAME = {
    CLASS_BG: "Background",
    CLASS_ELECTRON: "Electron",
    CLASS_MUON: "Muon",
    CLASS_MICHEL: "Michel",
    CLASS_BREMS: "Brems",
}

IGNORE_INDEX = 255

# Detector geometry

PARTITION_SEED = 12345
PARTITION_FRAC = 1.0
TRAIN_FRACTION = 1.0


THETA_DEG = {
    'Z':  0.0,
    'U': +35.71,
    'V': -35.71,
}

VBULK   = 1.60e-4   # cm/ns
T_TICK  = 500.0     # ns per tick
PITCH_Z = 0.479     # cm per wire
PITCH_U = 0.4667
PITCH_V = 0.4667
