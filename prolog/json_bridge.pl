:- module(json_bridge, [main/0]).

:- use_module(library(http/json)).
:- use_module('./query.pl', [all_candidates/2, goal_reached/2]).


main :-
    catch(run_request, Error, write_error(Error)).


run_request :-
    json_read_dict(current_input, Request),
    get_dict(facts, Request, FactDicts),
    get_dict(goal, Request, GoalDict),
    maplist(fact_dict_term, FactDicts, Facts),
    fact_dict_term(GoalDict, Goal),
    (   goal_reached(Facts, Goal)
    ->  GoalReached = true
    ;   GoalReached = false
    ),
    all_candidates(Facts, Candidates),
    maplist(candidate_dict, Candidates, CandidateDicts),
    json_write_dict(
        current_output,
        _{ok: true, goal_reached: GoalReached, candidates: CandidateDicts}
    ),
    nl.


write_error(Error) :-
    message_to_string(Error, Message),
    json_write_dict(current_output, _{ok: false, error: Message}),
    nl,
    halt(1).


fact_dict_term(Dict, Term) :-
    get_dict(pred, Dict, PredicateValue),
    json_atom(PredicateValue, Predicate),
    get_dict(args, Dict, JsonArgs),
    maplist(json_arg_term, JsonArgs, Args),
    Term =.. [Predicate | Args].


json_arg_term(Value, Term) :-
    is_list(Value), !,
    Value = [FunctorValue | JsonArgs],
    json_atom(FunctorValue, Functor),
    maplist(json_arg_term, JsonArgs, Args),
    Term =.. [Functor | Args].
json_arg_term(Value, Atom) :-
    json_atom(Value, Atom).


json_atom(Value, Atom) :-
    string(Value), !,
    atom_string(Atom, Value).
json_atom(Value, Value) :-
    atom(Value), !.
json_atom(Value, Value) :-
    number(Value), !.
json_atom(Value, _) :-
    throw(error(type_error(json_atom, Value), json_bridge)).


candidate_dict(candidate(Rule, Derived), Dict) :-
    atom_string(Rule, RuleString),
    term_fact_dict(Derived, DerivedDict),
    Dict = _{rule: RuleString, derived: DerivedDict}.


term_fact_dict(Term, Dict) :-
    compound_name_arguments(Term, Predicate, Args),
    atom_string(Predicate, PredicateString),
    maplist(term_json_arg, Args, JsonArgs),
    Dict = _{pred: PredicateString, args: JsonArgs}.


term_json_arg(Term, Json) :-
    compound(Term), !,
    compound_name_arguments(Term, Functor, Args),
    atom_string(Functor, FunctorString),
    maplist(term_json_arg, Args, JsonArgs),
    Json = [FunctorString | JsonArgs].
term_json_arg(Atom, String) :-
    atom(Atom), !,
    atom_string(Atom, String).
term_json_arg(Value, Value).
