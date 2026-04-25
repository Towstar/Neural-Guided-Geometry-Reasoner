:- module(query, [
    all_candidates/2,
    new_derivation/3,
    goal_reached/2,
    apply_derivation/3
]).

:- use_module('./rules.pl', [derive/3, canonical_fact/2]).


% --------------------------------------------------
% Check whether a goal is already present in the fact set
% --------------------------------------------------

goal_reached(Facts, Goal) :-
    canonical_fact(Goal, CanonGoal),
    memberchk(CanonGoal, Facts).


% --------------------------------------------------
% A derivation is "new" if it is derivable and not already in Facts
% --------------------------------------------------

new_derivation(Facts, Derived, Rule) :-
    derive(Facts, Derived, Rule),
    \+ memberchk(Derived, Facts).


% --------------------------------------------------
% Collect all unique candidate derivations
%
% Returns a list of:
%   candidate(Rule, Derived)
% --------------------------------------------------

all_candidates(Facts, Candidates) :-
    setof(
        candidate(Rule, Derived),
        new_derivation(Facts, Derived, Rule),
        Candidates
    ), !.

all_candidates(_, []).


% --------------------------------------------------
% Apply a derived fact by adding it to the fact set
% --------------------------------------------------

apply_derivation(Facts, Derived, NewFacts) :-
    canonical_fact(Derived, CanonDerived),
    (   memberchk(CanonDerived, Facts)
    ->  NewFacts = Facts
    ;   NewFacts = [CanonDerived | Facts]
    ).