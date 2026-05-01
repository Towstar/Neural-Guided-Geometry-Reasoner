:- module(query, [
    all_candidates/2,
    new_derivation/3,
    goal_reached/2,
    apply_derivation/3
]).

:- use_module('./rules.pl', [derive/3, canonical_fact/2]).


% --------------------------------------------------
% Normalize and deduplicate a fact set before querying
% --------------------------------------------------

canonicalize_facts(Facts, CanonFacts) :-
    maplist(canonical_fact, Facts, CanonFacts0),
    sort(CanonFacts0, CanonFacts).


% --------------------------------------------------
% Check whether a goal is already present in the fact set
% --------------------------------------------------

goal_reached(Facts, Goal) :-
    canonicalize_facts(Facts, CanonFacts),
    canonical_fact(Goal, CanonGoal),
    memberchk(CanonGoal, CanonFacts).


% --------------------------------------------------
% A derivation is "new" if it is derivable and not already in Facts
% --------------------------------------------------

new_derivation(Facts, Derived, Rule) :-
    canonicalize_facts(Facts, CanonFacts),
    derive(CanonFacts, Candidate, Rule),
    canonical_fact(Candidate, Derived),
    \+ memberchk(Derived, CanonFacts).


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
    canonicalize_facts(Facts, CanonFacts),
    canonical_fact(Derived, CanonDerived),
    (   memberchk(CanonDerived, CanonFacts)
    ->  NewFacts = CanonFacts
    ;   NewFacts = [CanonDerived | CanonFacts]
    ).
