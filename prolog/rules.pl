:- module(rules, [derive/3, canonical_fact/2]).

/*
This file contains helper predicates for canonicalizing unordered geometry objects as
well as geometry derivation rules.

*/

%-----------------------
% basic ordering helpers
%-----------------------

% Returns true if X = Y or X is before Y in the standard term order.
ordered_pair(X,Y,X,Y) :-
X @=< Y, !.
ordered_pair(X,Y,Y,X).

% Returns true if X = Y or X is before Y in the standard term order.
ordered_terms(X,Y,X,Y) :-
X @=< Y, !.
ordered_terms(X,Y,Y,X).

%-------------------------------------
% Canonicalization of structured terms
%-------------------------------------

canonical_term(segment(A,B), segment(X,Y)) :-
ordered_pair(A,B,X,Y).

canonical_term(line(A,B), line(X,Y)) :-
ordered_pair(A,B,X,Y).

canonical_term(angle(A,B,C), angle(A,B,C)).

canonical_term(T,T) :-
atomic(T).

%-------------------------------------
% Canonicalization of facts
%-------------------------------------

canonical_fact(midpoint(M,A,B), midpoint(M,X,Y)) :-
ordered_pair(A,B,X,Y).

canonical_fact(collinear(A,B,C), collinear(X,Y,Z)) :-
msort([A,B,C], [X,Y,Z]).

canonical_fact(between(A,B,C), between(A,B,C)).

canonical_fact(triangle(A,B,C), triangle(A,B,C)).

canonical_fact(on_perp_bisector(P,A,B), on_perp_bisector(P,X,Y)) :-
ordered_pair(A,B,X,Y).

canonical_fact(angle_bisector(P,A,B,C), angle_bisector(P,A,B,C)).

canonical_fact(is_isosceles(A,B,C), is_isosceles(A,B,C)).

canonical_fact(equal_length(S1, S2), equal_length(X1, X2)) :-
canonical_term(S1, C1),
canonical_term(S2, C2),
ordered_terms(C1, C2, X1, X2).

canonical_fact(parallel(L1, L2), parallel(X1, X2)) :-
canonical_term(L1, C1),
canonical_term(L2, C2),
ordered_terms(C1, C2, X1, X2).

canonical_fact(perpendicular(L1, L2), perpendicular(X1, X2)) :-
canonical_term(L1, C1),
canonical_term(L2, C2),
ordered_terms(C1, C2, X1, X2).

canonical_fact(equal_angle(A1, A2), equal_angle(X1, X2)) :-
canonical_term(A1, C1),
canonical_term(A2, C2),
ordered_terms(C1, C2, X1, X2).

canonical_fact(F, F).

has_fact(Facts, Fact) :- memberchk(Fact, Facts).

% midpoint(M,A,B) -> equal_length(segment(A,M), segment(M,B))
derive(Facts, Derived, midpoint_def_equal_segments) :-
    has_fact(Facts, midpoint(M, A, B)),
    canonical_fact(
        equal_length(segment(A, M), segment(M, B)),
        Derived
    ).

% midpoint(M,A,B) -> collinear(A,M,B)
derive(Facts, Derived, midpoint_def_collinear) :-
    has_fact(Facts, midpoint(M, A, B)),
    canonical_fact(
        collinear(A, M, B),
        Derived
    ).

% between(A,B,C) -> collinear(A,B,C)
derive(Facts, Derived, between_implies_collinear) :-
    has_fact(Facts, between(A, B, C)),
    canonical_fact(
        collinear(A, B, C),
        Derived
    ).

% equal_length(segment(A,M), segment(M,B)) and collinear(A,M,B) -> midpoint(M,A,B)
derive(Facts, Derived, midpoint_converse) :-
    has_fact(Facts, equal_length(segment(A, M), segment(M, B))),
    has_fact(Facts, collinear(A, M, B)),
    canonical_fact(
        midpoint(M, A, B),
        Derived
    ).

% midpoint(D,A,B) and midpoint(E,A,C) and triangle(A,B,C) -> parallel(line(D,E), line(B,C))
derive(Facts, Derived, midsegment_theorem) :-
    has_fact(Facts, triangle(A, B, C)),
    has_fact(Facts, midpoint(D, A, B)),
    has_fact(Facts, midpoint(E, A, C)),
    canonical_fact(
        parallel(line(D, E), line(B, C)),
        Derived
    ).

% triangle(A,B,C) and AB = AC -> angle ABC = angle BCA
derive(Facts, Derived, equal_legs_imply_base_angles) :-
    has_fact(Facts, triangle(A, B, C)),
    has_fact(Facts, equal_length(segment(A, B), segment(A, C))),
    canonical_fact(
        equal_angle(angle(A, B, C), angle(B, C, A)),
        Derived
    ).

% P on perpendicular bisector of AB -> PA = PB
derive(Facts, Derived, perpendicular_bisector_equidistance) :-
    has_fact(Facts, on_perp_bisector(P, A, B)),
    canonical_fact(
        equal_length(segment(P, A), segment(P, B)),
        Derived
    ).

% AP bisects angle BAC -> angle BAP = angle PAC
derive(Facts, Derived, angle_bisector_definition) :-
    has_fact(Facts, angle_bisector(P, A, B, C)),
    canonical_fact(
        equal_angle(angle(B, A, P), angle(P, A, C)),
        Derived
    ).

% transitivity of segment equality
derive(Facts, Derived, equal_length_transitivity) :-
    has_fact(Facts, equal_length(S1, S2)),
    has_fact(Facts, equal_length(S2, S3)),
    S1 \= S3,
    canonical_fact(
        equal_length(S1, S3),
        Derived
    ).

% triangle(A,B,C) and AB = AC -> is_isosceles(A,B,C)
derive(Facts, Derived, equal_legs_imply_isosceles) :-
    has_fact(Facts, triangle(A, B, C)),
    has_fact(Facts, equal_length(segment(A, B), segment(A, C))),
    canonical_fact(
        is_isosceles(A, B, C),
        Derived
    ).
