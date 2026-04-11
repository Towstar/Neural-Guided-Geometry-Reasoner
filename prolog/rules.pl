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

