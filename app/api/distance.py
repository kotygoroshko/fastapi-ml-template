from fastapi import APIRouter
from enum import Enum
import textdistance

distance_router = APIRouter()

# class DistanceAlgorythmName(Enum):
#     Hamming             = ('Hamming', textdistance.hamming)
#     Mlipns              = ('Mlipns',textdistance.mlipns)
#     Levenshtein         = ('Levenshtein',textdistance.levenshtein)
#     DamerauLevenshtein  = ('DamerauLevenshtein',textdistance.damerau_levenshtein)
#     JaroWinkler         = ('Jaro-Winkler',textdistance.jaro_winkler)
#     StrCmp95            = ('StrCmp95',textdistance.strcmp95)
#     NeedlemanWunsch     = ('Needleman-Wunsch',textdistance.needleman_wunsch)
#     Gotoh               = ('Gotoh', textdistance.gotoh)
#     SmithWaterman       = ('Smith-Waterman', textdistance.smith_waterman)

#     def __init__(self, algorythm_name, distance_function):
#         self.algorythm_name = algorythm_name
#         self.distance_function = distance_function   

#     @property
#     def distance(self,str1,str2):
#         return self.distance

class DistanceAlgorythmName(Enum):
    Hamming             = "Hamming"
    Mlipns              = "Mlipns"
    Levenshtein         = "Levenshtein"
    DamerauLevenshtein  = "DamerauLevenshtein"
    JaroWinkler         = "Jaro-Winkler"
    StrCmp95            = "StrCmp95"
    NeedlemanWunsch     = "Needleman-Wunsch"
    Gotoh               = "Gotoh"
    SmithWaterman       = "Smith-Waterman"  

actions = {
    DistanceAlgorythmName.Hamming             : textdistance.hamming,
    DistanceAlgorythmName.Mlipns              :textdistance.mlipns,
    DistanceAlgorythmName.Levenshtein         :textdistance.levenshtein,
    DistanceAlgorythmName.DamerauLevenshtein  :textdistance.damerau_levenshtein,
    DistanceAlgorythmName.JaroWinkler         :textdistance.jaro_winkler,
    DistanceAlgorythmName.StrCmp95            :textdistance.strcmp95,
    DistanceAlgorythmName.NeedlemanWunsch     :textdistance.needleman_wunsch,
    DistanceAlgorythmName.Gotoh               :textdistance.gotoh,
    DistanceAlgorythmName.SmithWaterman       :textdistance.smith_waterman,
}

@distance_router.get('/distance/{distance_algorythm}', status_code=200)
def distance(distance_algorythm: DistanceAlgorythmName, str1: str, str2: str) -> dict:
    """
    Calculate distance
    """ 
    return {"Algorythm selected": distance_algorythm, "Str1": str1, "Str2": str2, "distance": actions[distance_algorythm](str1,str2)}