from typing import List, Tuple, Dict, Deque

from numpy import array

# StrPresentation = array(List[List[List[bytes]]])

Action = int

StrPresentation = List[List[List[bytes]]]  # because if wrapped in array, NodeType doesnt accept it well

StateEncoding = List[List[List[int]]]
ActionEncoding = List[List[List[List[Action]]]]

NodeType = Dict[Tuple[StateEncoding, StrPresentation], int]
Nsa = Dict[Tuple[StrPresentation, int], int]
Qsa = Dict[Tuple[StrPresentation, Action], float]

Pi = List[float]
V = List[float]
Ps = Dict[StrPresentation, Pi]
Ns = Ps

# CoachEpisode = List[Tuple[ActionEncoding, Pi, V]]
CoachEpisode = List[Tuple[ActionEncoding, Pi, V]]

CanonicalBoard = array(StateEncoding)

ValidMoves = array(List[int])

TrainExamples = List[Tuple[ActionEncoding, int, List[int], None]]
TrainExamplesHistory = List[Deque[CoachEpisode]]

Sym = List[Tuple[ActionEncoding, List[int]]]
