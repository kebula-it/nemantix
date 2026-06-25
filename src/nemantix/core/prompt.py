PROLOGUE = """
You are working with NXS, a DSL made of structured blocks.

CORE CONCEPTS
- An NXS file can include other NXS files, define reusable schemas with `frame`, declare tools with `toolset`, define executable units with `action`, and define a runnable workflow with `deliberate`.
- An `action` is like a function: it can declare inputs (`in:`), outputs (`out:`), and a `body:` with logic such as expressions, calls, conditionals, loops, and control flow.
- A `deliberate` is the main executable workflow: it includes a `when` trigger, `guidelines`, a `plan` and in some special case some private `action`s.
- The `plan` of a `deliberate` is the orchestration logic that aggregates and coordinates multiple `action` call to achieve the desired outcome following the `guidelines`. Like an `action`, it can declare inputs (`in:`), outputs (`out:`), and a `body:`.
"""

CODING_SYSTEM_PROMPT = (
    PROLOGUE
    + """
PROMPTS INSIDE CODE (CRITICAL)
NXS may contain natural-language prompts delimited as:
- inline prompt: `>> ... <<` on the same line
- block prompt: `>>> ... <<<` across multiple lines

These prompts may appear anywhere in the code.

TWO TYPES OF PROMPTS
1) COMPLETION PROMPTS
- They describe behavior that must be implemented in valid NXS code.
- You must implement their meaning, but keep the original prompt text present exactly as written for traceability.
- Typically found alone in the lines of the body or as conditions in if statements and loops.

2) VERIFICATION PROMPTS
- Short semantic hints used only as checks/constraints during execution. 
- They don't describe an implementable behaviour, but they are descriptions.
- They must remain exactly as written and must not be replaced or rewritten.

RULES
- Always output valid NXS syntax.
- For completion prompts, preserve the original prompt verbatim and add the corresponding NXS implementation.
- For verification prompts, leave them untouched.
- Do not invent constructs outside NXS.

REFERENCE EXAMPLE (NXS + comments)
```
{Demo}
@intent.goal: "normalization and preparation"  # intentable prefix on an action (label + metadata)
# each action declaration needs a name and a microprompt that explains the purpose of the action
action NormalizeInput >> Demonstrate prompts, variables, calls, structures, and metadata. <<:
    in:
        raw_text (required) >> text input, may contain extra spaces <<
        users (default [()]) >> optional input user list <<
    __in

    out:
        clean_text >> normalized text <<
        prepared_users >> prepared user list <<
        trace >> trace string <<
    __out

    body:
        # PROMPTS INSIDE CODE:
        # - Inline prompt: `>> ... <<` (single line, explicitly closed)
        # - Block prompt:  `>>> ... <<<` (multiline, explicitly closed)
        # Prompts may be completion prompts (implement their meaning, while preserving the original text)
        # or verification prompts (must remain unchanged).

        # COMPLETION PROMPT:
        # - This prompt must remain exactly as written for traceability.
        # - Its meaning must also be implemented in valid NXS code.
        >> Normalize the input text and remove extra surrounding spaces. <<

        # VARIABLES:
        # - Variables are in square brackets: [x]
        # - Accessors use `:` for keys/indices/expressions: [m:key], [arr:0], [x:<expr>]
        # - You can also embed a micro-prompt in a variable token (prompt runs until `]`) for verification
        [ [clean_text >> cleaned text <<] = [raw_text] ?? "" ]  # EXPRESSIONS: outer [] wraps an expression; `??` is fallback
        # - Keywords (e.g., when, from, use, as, with, ...) are forbidden as variable names!

        # DO CALLS:
        # - Inline: `do optional_callable_type q.name using <args> producing <args> >> note or details`
        # - Block:  `do optional_callable_type q.name: ... __do`
        # Here we show the inline form:
        # - args are assignment expressions ( [arg_name]=[variable] or [arg_name]=value ).
        # - output is an assignment with a variable [[variable]] or a list of variables [ [var1], [var2]]
        # - optional_callable_type is the type of callable (one of action or tool)
        # There are built-in functions (to be called without any optional_callable_type) that can be invoked using the do clause; see below for the full list.
        # NOTE: calling a deliberate by name is FORBIDDEN!
        do tool trim using [ [text] = [clean_text], [max_char] = 0 ] producing [ [clean_text] ] >> producing the trimmed string
        do print using [ [text] = "text cleaned" ] >> debug print (example of built-in function)
        do action action_name using [ [value] = [result] ] producing [ [action_result] ] >> example call of action
        # `using` and `producing` clause are optional if there are no inputs or no outputs
        do action list_available_pizza 
        do action create_menu producing [ [menu] ]
        do action delete_menu using [ [menu] ]

        # LISTS + KEY:VALUE:
        # - List literals use parentheses: (a, b, c)
        # - Key:value entries: ("k": v) or (k: v)
        [ [user] = ( name: "alice", age: 30, tags: ("dev", "ops") ) ]
        [ [user:student] = false ]
        [ [user:worker] = true ]
        [ [prepared_users] = [users] ?? ([user]) ]

        # STRUCTURE / FRAME APPLY:
        # - Apply a frame to a list literal with {Qualified.Name} (before or after the list).
        # Prefix application requires the Structured Collection to conform to the frame definition;
        # Postfix application interprets the structure in the context of the frame, even if it is partial or incomplete.
        [ [constrained] = {PERSON}( name: "alice", age: 30, tags: ("dev", "ops") ) ]
        [ [partial] = ( name: "bob", tags: ("ml", "ai") ){PERSON} ]

        # META EXPRESSIONS:
        # - {{Name@qualified.path}} can appear in expressions/metadata to inject runtime/meta values.
        [ [trace] = "trace=" | {{Run@demo.two_action}} ]  # `|` is concatenation

        return [ [clean_text], [prepared_users], [trace] ] # CONTROL FLOW: return with multiple values
    __body
__action

{Demo}
@semantics: "analysis-and-reporting"  # intentable prefix on an action (label + metadata)
# each action declaration needs a name and a microprompt that explains the purpose of the action
action AnalyzeUsers >> Demonstrate conditionals, loops, semantic comparisons, and final output building. <<:
    in:
        clean_text (required) >> normalized text <<
        users (default [()]) >> list of users <<
        threshold (default [2]) >> the threshold, int >= 0 <<
        trace (default [""]) >> trace string <<
    __in

    out:
        report_text >> final output string, non-empty <<   # example of VERIFICATION PROMPT in output declaration
    __out

    body:
        # CONDITIONALS:
        # - if/elif/else blocks end with `__if`
        # - Conditions can be an expression or a prompted condition that must be completed
        if [ [threshold] <= 0 ]:
            [ [threshold] = 1 ]
        __if

        # LOOPS (`repeat`):
        # Variants include:
        # - intent-driven: `repeat >>> intent <<<: ... __repeat` that must be completed
        # - each/as:       `repeat each <iterable> as <i>,<item> where <cond>: ... __repeat`
        # - times:         `repeat <n> times as <i>: ... __repeat`
        # - while/until:   `repeat while/until <cond> max <n>: ... __repeat`
        # We show ONE variant here: `repeat each ... where ...`
        [ [stats] = ("user_count": 0, "tag_count": 0) ]

        repeat each [users] as [i], [u] :
            [ [stats] = ("user_count": ([stats:user_count] + 1), "tag_count": [stats:tag_count]) ]

            # SEMANTIC COMPARISONS:
            # It is used only when you need a semantic comparison via embedding of string.
            # - Similarity: a ~ b  (optionally qualified: a ~ close ~ b, or with a numeric qualifier)
            # - Inclusion:  a ~> b and reverse inclusion: a <~ b
            if [ [u:role] ~ close ~ "admin" ]:
                [ [stats] = ("user_count": [stats:user_count], "tag_count": ([stats:tag_count] + 2)) ]
                continue  # CONTROL FLOW: continue
            __if

            # Demonstrate break via a small inner condition (still within the loop)
            if [ [stats:user_count] >= [threshold] ]:
                break  # CONTROL FLOW: break
            __if
        __repeat

        # BUILD FINAL OUTPUT STRING
        [ [report_text] =
           "text=" | [clean_text] | ", users=" | [stats:user_count] | ", tags=" | [stats:tag_count] | ", " | [trace]
        ]

        # STRING USE AND F-STRING RULES
        # - strings are always enclosed in double quotes "string text". Single quotes form 'string text' IS NOT ALLOWED! 
        # Example:
        # [ [wrong_syntax] = 'this is a string']   # THIS IS NOT ALLOWED!!!
        # [ [ right_syntax ] = "this is a string" ] # THIS IS THE RIGHT SYNTAX
        # [ [example] = "I can use 'single quotes' inside a string"]  # you can use single quotes as a content of a string

        # - if a string contains a double quote, escape it (\")
        # - you can use the f-string to format a string with the value of a variable, example:
        [ [user_stat] = "users number is [stats:user_count]" ]  # will be formatted to "users number is 3" where 3 is the value of stats:user_count
        # - REMEMBER: Square brackets in nxs are special symbols. If you need square brackets in the text of a string, you need to escape them (with ONE backslash for each bracket), example:
        [ [example] = "This is a square bracket symbol: \\[." ]


        return [report_text] # CONTROL FLOW: return with expression. For multiple values: [[v1], [v2]]
    __body
__action

deliberate DemoMain when >>> Run this deliberate when the user asks for a two-action syntax showcase. <<<:
    # deliberate declaration needs a name and a short micro-prompt showing its purpose

    guidelines:
        # PROMPTS (LANGUAGE TOKENS):
        # - Inline prompt: `>> ... <<` (single line, explicitly closed)
        # - Block prompt:  `>>> ... <<<` (multiline, explicitly closed)
        # Some prompts are completion prompts (implement their meaning while preserving the original text),
        # others are verification prompts (MUST stay unchanged).
        >> Keep verification prompts unchanged; implement completion prompts without removing their original text. <<
    __guidelines

    plan:
        # The plan aggregates and coordinates multiple actions to achieve the desired outcome
        # while following the deliberate guidelines.

        in:
            raw_text (required) >> text input, may contain extra spaces <<
            users (default [()]) >> list of users to analyze <<
            threshold (default [2]) >> the threshold, int >= 0 <<
        __in

        out:
            report_text >> final output string, non-empty <<
        __out

        body:
            [ [clean_text] = "" ]
            [ [prepared_users] = () ]
            [ [trace] = "" ]

            # DO CALLS:
            # Here the plan uses both actions.
            # First we show the inline form.
            do action NormalizeInput using [ [raw_text] = [raw_text], [users] = [users] ] producing [ [clean_text], [prepared_users], [trace] ] >> normalize input and prepare user data

            # Then we show the block form.
            do action AnalyzeUsers:
                using [ [clean_text] = [clean_text], [users] = [prepared_users], [threshold] = [threshold], [trace] = [trace] ]
                producing [ [report_text] ]
                >> build the final report
            __do
            
            # Inline version with frame application to the result with 'as {frame_name}'
            do action extract_info using [ [text] = [raw_text] ] producing [ [summary] ] as {summary_frame}

            return [report_text] # CONTROL FLOW: return with expression
        __body
    __plan
__deliberate
```
BUILT-IN FUNCTIONS LIST:
- print(*args, **kwargs): Wrapper around Python print, passing through arguments (and kwargs if provided).
- coalesce(*args, **kwargs): Returns the first argument that is not None; otherwise returns None.
- exists(x): Checks whether x is not None.
- size(*args): Returns a “size” depending on type (e.g., len for strings/Struct; 0/1 for Opaque; 1/0 for DocRef leaf/non-leaf; otherwise 0).
- type(x): Returns a string describing the type of the input variable. Possible types are: none; num (for integers and floats); 
str (strings); bool (booleans); struct (for Nemantix structures, i.e., the ones within "(...)"); doc (for Nemantix Docref objects);
and opaque (for Nemantix Opaque objects).
- substring(x, start=0, end: int): Converts x to string and returns the slice [start:end] (safe defaults if inputs are invalid).
- to_num(x): Explicit numeric conversion (handles numbers, booleans, numeric strings); returns 0 on failure.
- to_bool(x): Explicit boolean conversion (handles booleans, numbers, strings like "true"/"false"/"none"); defaults to False.
- to_str(x): Explicit string conversion (ALWAYS use it when you use concatenations with non-string objects).
- num(x): Soft numeric conversion: returns None for None or complex/container-like types; otherwise uses to_num.
- bool(x): Soft boolean conversion: returns None for None, uses type-specific rules for Struct/DocRef/Opaque, otherwise to_bool.
- str(x): Soft string conversion: returns None for None, otherwise uses to_str.
- sin(x): Computes sine of x after converting it to a number.
- cos(x): Computes cosine of x after converting it to a number.
- sqrt(x): Computes square root of x after converting it to a number.
- llm(prompt, **kwargs): Calls llm.invoke(prompt, **kwargs) and returns the response text as a Python string.
- retrieve: Used to retrieve information from the knowledge base. Calls retrieve(query: str, top_k: int = 5, min_score: float = 0.6, 
doc_types: list | str | None, content_types: list | str | None, metadata: list | str | None).
doc_types, content_types, metadata are filters.
Allowed doc_types: book, article, manual.
Allowed content_types: text, table, image.
Allowed metadata: character, place, rule, date.
- expand: Used to retrieve information from the knowledge base. Given a node id, retrieves its subnodes. Calls expand(node_id: str).
- extend: Used to retrieve information from the knowledge base. Given a node id, retrieves its siblings. Calls extend(node_id: str).
- generalize: Used to retrieve information from the knowledge base. Given a node id, retrieves its parent node. Calls expand(node_id: str).
"""
)

####################### ACTION PROMPTS #######################
COMPILATION_ACTION = """
YOUR TASK
Complete the NXS `action` given below.

WHAT YOU MUST PRODUCE
- Output ONLY one valid NXS `action` block, ending with `__action`.
- Do NOT output explanations, markdown, or any text outside the NXS code.

CONTEXT
- The input always already contains a shallow implementation of the action.
- Your job is to complete it, not replace the whole structure.

REQUIRED STRUCTURE
- Preserve the existing action header.
- Preserve the existing `in:`, `out:` blocks, complete them only if needed.
- You have to add the `@intent.goal` annotation, which must describe the semantic purpose of the action like a short docstring.

RULES
{rules}

ACTION TO COMPLETE
{action_nxs}
"""

DRAFT_ACTION_RULES = """- Complete `in:`, `out:`, and `body:` only with what can be validly inferred from the available code, prompts, micro-prompts, and guidelines.
- The produced action may be partial if some implementation details depend on information that will only be provided later by the user request.
- You must decide what can already be encoded now and what must remain deferred to a later coding pass.
- Encode now everything that is structurally clear, semantically grounded, and already implied by the available information.
- Do not invent missing business logic, user data, constants, branches, or assumptions that are not supported by the current input.
- Leave unresolved only the parts that you think require future user-provided information.
- Even if the implementation is partial, the output must remain valid NXS and preserve the full action structure.
- The current `body:` should prepare the action for later completion rather than guessing missing details.
- Follow the rules given about the prompts: Verification micro-prompts must remain exactly unchanged; completion prompts micro-prompts must be implemented (in this case, when enough information is available).
- Use only valid NXS constructs.
- You can use and call the tools and action described in the section below. Respect EXACTLY the described inputs and outputs of the tools and actions.
"""

COMPLETE_ACTION_RULES = """- Complete empty or partial `in:`, `out:`, and `body:` blocks using the available completion prompts, micro-prompts.
- Do not rename, remove, or alter declared inputs or outputs unless this is strictly necessary to make the action valid.
- The `body:` must be consistent with the declared `in:` and `out:`.
- Follow the rules given about the prompts: Verification prompts must remain exactly unchanged; completion micro-prompts must be implemented as valid NXS code.
- When a completion prompt is present, keep its original text exactly as written for traceability, and implement its meaning in NXS.
- Use only valid NXS constructs.
- Generate complete, coherent, and minimal NXS code.
- You can call the tools and action described in the section below. Respect EXACTLY the described inputs and outputs of the tools and actions.
- You can use the available `frame`s
"""

EVALUATE_ACTION_RULES = """- The input action must be treated as a draft implementation that should be completed as far as possible.
- Complete `in:`, `out:`, and `body:` using everything that can be validly inferred from the available code, micro-prompts, and additional context.
- Prefer completing the action now whenever the structure, semantics, or intended behavior are already clear enough.
- If you are sure that some details genuinely depend on information that will only be available later from the user request, leave only those parts unresolved.
- Do not invent unsupported business logic, user data, constants, branches, or assumptions that are not grounded in the current input.
- Don't be lazy and complete the code if you have enough information.
- Preserve declared inputs and outputs, and do not rename, remove, or alter them unless this is strictly necessary to make the action valid.
- If `in:` or `out:` is partial or shallow, complete it consistently with the action semantics and the available information.
- The `body:` must be consistent with the declared `in:` and `out:`, and should implement as much valid behavior as can be determined.
- If full implementation is not yet possible, the current `body:` should still provide the best valid partial coding, preparing the action for a later completion pass without guessing missing details.
- Follow the rules about prompts:
  - verification micro-prompts must remain exactly unchanged;
  - completion micro-prompts must be implemented as valid NXS code whenever enough information is available;
  - when a completion prompt is present, keep its original text exactly as written for traceability, and implement its meaning in NXS.
- Even if the implementation remains partial, the output must still be valid NXS and preserve the full action structure.
- Use only valid NXS constructs.
- Generate coherent, minimal, and as-complete-as-possible NXS code of the whole action block.
- You can use and call the tools, actions, and frames described in the additional context. Respect EXACTLY the described inputs and outputs of the tools and actions.
"""

CODING_ADDITIONAL_INFO = """For the generation of the code you can use the following additional information:
TOOLS YOU CAN USE (name and description)
{tools}
OTHER ACTIONS YOU CAN USE:
{actions}
FRAMES
{frames}
ENV VARIABLES
You can use variables from the environment with the syntax [ENV:var_name].
The available var_name are:
{ENV_vars}
"""

USER_REQUEST = """
USER REQUEST
Use also the following user request, if it is useful, to complete the code:
{user_request}
"""

##############################################################
##################### DELIBERATE PROMPTS #####################
COMPILATION_DELIBERATE = """
YOUR TASK
Complete the `plan` of the NXS `deliberate` given below.

WHAT YOU MUST PRODUCE
- Output ONLY one valid NXS `plan` block, from `plan:` to `__plan`.
- Do NOT output the full `deliberate`.
- Do NOT output explanations, markdown, or any text outside the NXS code.

CONTEXT
- The deliberate header, `when`, and `guidelines` are already defined.
- The `plan` already contains `in:`, `out:`, and `body:` blocks, and they may be filled, partially filled, or empty.
- Your job is to complete the existing `plan`, not to rewrite the whole deliberate.

REQUIRED STRUCTURE
- Preserve and complete `plan.in:`, `plan.out:`, and `plan.body:` as needed.
- The `plan` must behave as the orchestration unit of the deliberate.
- The `plan` must define a coherent execution flow aligned with the deliberate guidelines.
- You have to add the `@intent.goal` annotation, which must be a short description of your implementation of the plan.

RULES
{rules}

DELIBERATE TO COMPLETE
{deliberate_nxs}
"""

DRAFT_DELIBERATE_RULES = """- Complete the `plan` only with what can be validly inferred from the available code, micro-prompts, guidelines, and deliberate intent.
- Complete `plan.in:`, `plan.out:`, and `plan.body:` only where enough information is available.
- The produced `plan` may be partial if some orchestration details depend on information that will only be provided later by the user request.
- You must decide what can already be encoded now and what must remain deferred to a later coding pass.
- Encode now everything that is structurally clear, semantically grounded, and already implied by the available information.
- Do not invent missing business logic, user data, constants, branches, sequencing rules, or assumptions that are not supported by the current input.
- Leave unresolved only the parts that genuinely require future user-provided information.
- Even if the implementation is partial, the output must remain valid NXS and preserve the full `plan` structure.
- The `plan` must be compiled so that it creates a coherent execution plan consistent with the `guidelines`, aggregating the appropriate actions and tools in valid NXS.
- The `plan` should use `do` calls to actions and tools when their role is already implied by the available information.
- Follow the rules given about the prompts: verification micro-prompts must remain exactly unchanged; completion micro-prompts must be implemented when enough information is available.
- If a completion prompt cannot yet be fully implemented because it depends on future user input, preserve it exactly and encode only the portion that can already be determined.
- Use only valid NXS constructs.
- You can use and call the tools, actions, and frames described in the section below. Respect EXACTLY the described inputs and outputs of the tools and actions.
"""

COMPLETE_DELIBERATE_RULES = """- Complete empty or partial `plan.in:`, `plan.out:`, and `plan.body:` blocks using the available completion prompts, micro-prompts, and guidelines.
- Do not rename, remove, or alter declared plan inputs or outputs unless this is strictly necessary to make the `plan` valid.
- The `plan.body:` must be consistent with the declared `plan.in:` and `plan.out:`.
- The `plan` must be compiled so that it defines a coherent execution plan aligned with the deliberate `guidelines`, aggregating the needed actions and tools in valid NXS.
- The orchestration in `plan.body:` must be explicit and valid: sequence calls coherently, pass inputs correctly, and produce outputs consistent with the plan contract.
- Follow the rules given about the prompts: verification micro-prompts must remain exactly unchanged; completion micro-prompts must be implemented as valid NXS code.
- When a completion prompt is present, keep its original text exactly as written for traceability, and implement its meaning in NXS.
- Use only valid NXS constructs.
- Generate complete, coherent, and minimal NXS code.
- You can call the tools and actions described in the section below. Respect EXACTLY the described inputs and outputs of the tools and actions.
- You can use the available `frame`s.
"""

EVALUATE_DELIBERATE_RULES = """- The input `plan` must be treated as a draft implementation that should be completed as far as possible.
- Complete `plan.in:`, `plan.out:`, and `plan.body:` using everything that can be validly inferred from the available code, micro-prompts, guidelines, deliberate intent, and additional context.
- Prefer completing the plan now whenever the structure, orchestration, or intended behavior are already clear enough.
- If you think some orchestration details genuinely depend on information that will only be available later from the user request, leave only those parts unresolved.
- Do not invent unsupported business logic, user data, constants, branches, sequencing rules, or assumptions that are not grounded in the current input.
- Don't be lazy and complete the code if you have enough information.
- Preserve declared plan inputs and outputs, and do not rename, remove, or alter them unless this is strictly necessary to make the `plan` valid.
- If `plan.in:` or `plan.out:` is partial or shallow, complete it consistently with the plan semantics and the available information.
- The `plan.body:` must be consistent with the declared `plan.in:` and `plan.out:`, and should implement as much valid orchestration as can already be determined.
- The `plan` must define a coherent execution flow aligned with the deliberate `guidelines`, aggregating the needed actions and tools in valid NXS.
- The orchestration in `plan.body:` should be explicit and valid: sequence calls coherently, pass inputs correctly, and produce outputs consistent with the plan contract.
- Prefer using `do` calls to actions and tools whenever their role is already implied by the available information.
- If full implementation is not yet possible, the current `plan.body:` should still provide the best valid partial orchestration, preparing the plan for a later completion pass without guessing missing details.
- Follow the rules about prompts:
  - verification micro-prompts must remain exactly unchanged;
  - completion micro-prompts must be implemented as valid NXS code whenever enough information is available;
  - when a completion prompt is present, keep its original text exactly as written for traceability, and implement its meaning in NXS.
- Even if the implementation remains partial, the output must still be valid NXS and preserve the full `plan` structure.
- Use only valid NXS constructs.
- Generate coherent, minimal, and as-complete-as-possible NXS code.
- You can use and call the tools, actions, and frames described in the additional context. Respect EXACTLY the described inputs and outputs of the tools and actions.
"""

CODING_DELIBERATE_ADDITIONAL_INFO = (
    CODING_ADDITIONAL_INFO
    + """
OTHER DELIBERATES
{deliberates}

KNOWLEDGE_BASE CONTEXT
{knowledge_base}
"""
)

CODING_ACTION_ADDITIONAL_INFO = (
    CODING_ADDITIONAL_INFO
    + """
KNOWLEDGE_BASE CONTEXT
{knowledge_base}
"""
)
##############################################################
################ DELIBERATE WITH BREAKDOWN ###################
COMPILATION_DELIBERATE_BREAKDOWN = """
YOUR TASK
Complete the `plan` of the NXS `deliberate` given below and generate the necessary `action` blocks required for its implementation.

WHAT YOU MUST PRODUCE
- Output ONLY valid NXS code.
- The output must consist of:
  1. One or more `action` blocks (as needed for implementation).
  2. Followed by exactly one `plan` block, from `plan:` to `__plan`.
- The `plan` must come AFTER all `action` blocks.
- Do NOT output the full `deliberate`.
- Do NOT output explanations, markdown, or any text outside the NXS code.

CONTEXT
- The deliberate header, `when`, and `guidelines` are already defined.
- The `plan` may already contains `in:`, `out:`, and `body:` blocks, and they may be filled, partially filled, or empty.
- Your job is to complete the existing `plan`, not to rewrite the whole deliberate.

REQUIRED STRUCTURE
- Actions:
  - Define any required `action` blocks outside the plan.
  - Each action should represent a concrete implementation step.
- Plan:
  - Preserve and complete `plan.in:`, `plan.out:`, and `plan.body:` as needed.
  - The `plan` must act ONLY as an orchestration/aggregation layer.
  - The `plan.body` must reference and compose the previously defined actions.
  - The plan must NOT contain low-level implementation details.
  - The plan must define a coherent execution flow aligned with the deliberate guidelines.
  - You must add the `@intent.goal` annotation to each action, which must be a short description of your implementation (like a short docstring).

RULES
{rules}

DELIBERATE TO COMPLETE
{deliberate_nxs}
"""
DRAFT_DELIBERATE_BREAKDOWN_RULES = """- Complete the `deliberate` only with what can be validly inferred from the available code, micro-prompts, guidelines, and deliberate intent.
- If the deliberate already contains `action` blocks before the `plan`, treat them as available and reusable in the `plan`.
- Reuse existing actions first, including actions already inside the deliberate and actions provided in the additional context.
- Add new `action` blocks before the `plan` only if the available actions are not sufficient to satisfy the request.
- Any new action must be valid NXS and directly useful for the `plan`.
- The result may remain partial if some details depend on future user input.
- Remember to always generate the plan: its the main logic of the deliberate.
- Encode now everything that is already clear; do not invent unsupported logic, data, constants, branches, or sequencing rules.
- Leave unresolved only what genuinely requires future user-provided information.
- The `plan` must define a coherent execution flow consistent with the `guidelines`, aggregating actions and tools in valid NXS.
- Follow the rules about prompts:
  - verification micro-prompts must remain exactly unchanged;
  - completion micro-prompts must be implemented as valid NXS code whenever enough information is available;
  - when a completion prompt is present, keep its original text exactly as written for traceability, and implement its meaning in NXS.
- Use only valid NXS constructs.
- You can use and call (with `do`) the tools, actions, and frames described in the section below. Respect EXACTLY the described inputs and outputs of the tools and actions.
"""
COMPLETE_DELIBERATE_BREAKDOWN_RULES = """- Complete empty or partial `plan.in:`, `plan.out:`, and `plan.body:` using the available completion prompts, micro-prompts, and guidelines.
- If the deliberate already contains `action` blocks before the `plan`, treat them as available and reusable in the `plan`.
- Reuse existing actions first, including actions already inside the deliberate and actions provided in the additional context.
- Add new `action` blocks before the `plan` only if the available actions are not sufficient.
- Any new action must be minimal, valid, and directly useful for the deliberate flow.
- Do not rename, remove, or alter declared plan inputs or outputs unless strictly necessary to make the `plan` valid.
- Remember to always generate the plan: its the main logic of the deliberate.
- `plan.body:` must be consistent with `plan.in:` and `plan.out:`.
- The `plan` must define a coherent execution flow consistent with the `guidelines`, aggregating actions and tools in valid NXS.
- Follow the rules about prompts:
  - verification micro-prompts must remain exactly unchanged;
  - completion micro-prompts must be implemented as valid NXS code;
  - when a completion prompt is present, keep its original text exactly as written for traceability, and implement its meaning in NXS.
- Use only valid NXS constructs.
- You can use and call (with `do`) the tools, actions, and frames described in the section below. Respect EXACTLY the described inputs and outputs of the tools and actions.
"""
EVALUATE_DELIBERATE_BREAKDOWN_RULES = """- Treat the input `deliberate` as a draft that should be completed as far as possible.
- If the deliberate already contains `action` blocks before the `plan`, treat them as available and reusable in the `plan`.
- Reuse existing actions first, including actions already inside the deliberate and actions provided in the additional context.
- Add new `action` blocks before the `plan` only if the available actions are not sufficient.
- Any new action must be minimal, valid, and semantically grounded.
- Complete as much as possible now; leave unresolved only what genuinely depends on future user input.
- Do not invent unsupported logic, data, constants, branches, or sequencing rules.
- Preserve declared plan inputs and outputs unless changing them is strictly necessary to make the `plan` valid.
- Remember to always generate the plan: its the main logic of the deliberate.
- If `plan.in:` or `plan.out:` is partial, complete it consistently with the plan semantics and the available information.
- `plan.body:` must be consistent with `plan.in:` and `plan.out:` and should implement as much valid orchestration as can already be determined.
- The `plan` must define a coherent execution flow aligned with the `guidelines`, aggregating the needed actions and tools in valid NXS.
- Follow the rules about prompts:
  - verification micro-prompts must remain exactly unchanged;
  - completion micro-prompts must be implemented as valid NXS code whenever enough information is available;
  - when a completion prompt is present, keep its original text exactly as written for traceability, and implement its meaning in NXS.
- Even if partial, the output must remain valid NXS and preserve the full deliberate structure.
- Use only valid NXS constructs.
- You can use and call (with `do`) the tools, actions, and frames described in the section below. Respect EXACTLY the described inputs and outputs of the tools and actions.
"""

#######################################################################
#########################  TOOL GENERATION  ###########################
GEN_TOOLSET_PROMPT = """
You are an expert Python code generator.
Generate a Python class based on the provided name, description, and custom DSL syntax.
Strictly follow these rules:
    1. The class must be named exactly as requested in the 'Name' field and inherit
       from `Toolset`.
    2. Include a comprehensive class-level docstring summarizing its purpose based
       on the provided description. DO NOT mention the custom syntax, the DSL, or
       how the class was generated in the docstrings.
    3. Analyze the 'Import Statement'. The values inside the `with [...]` clause
       dictate the default parameters and values the `__init__` method must accept.
       Ensure `__init__` sets these as instance variables with accurate Python type
       hints.\n
    4. Analyze the `use ...` clause in the 'Import Statement'. The method you create
       to fulfill the user's description MUST be named exactly as specified after the
       `use` keyword. Decorate this method with `@tool`.
    5. Analyze the 'Usage Statements'. The values inside the `using [...]` clause
       dictate the specific arguments passed to the `@tool` method at runtime.
       Construct the parameters of the `@tool` method to accept these arguments,
       using accurate Python type hints based on the inferred data types in the list.
    6. The `@tool` method must also include a detailed docstring explaining its
       functionality, its arguments (Args:), and its return value (Returns:). Keep
       the docstring focused purely on the tool's usage.
    7. DO NOT write any import statements for `Toolset` or `tool`. Assume they are
       already injected into the global namespace.
    8. Output ONLY valid, executable Python code.
    9. NEVER wrap the code in markdown blocks (e.g., no ```python or ```).
       No explanations.
    10. If 'Previous Code' and an error are provided, use the previous code as your
        baseline. Correct ONLY the logic or syntax that caused the error, preserving
        the valid parts of the structure.
"""

####################################################################################

FIX_GENERATION = "The code you generated gave an error during the parsing. Fix the error and rewrite the full corrected code and nothing else. This is the error:"

INTENT_PROMPT = (
    PROLOGUE
    + """
For each `action` block in the following pseudo code, produce a brief doc string.
Produce a JSON with the name of the action and the docstring. Output example:
{"write_file": "Writes the given string to the specified file.",
"read_file": "Reads the given the specified file.",
}"""
)

# internally used by Executor/Interpreter
DELIBERATE_SELECTION_PROMPT = """I will now give you a dictionary describing the semantics of some pseudo-code block called 'deliberates'.
    [[delib_sem_map]]
    "{}"
    find the name of the deliberate statement that best answers the
    following user request:
    [[user request]]
    "{}"
    lastly, return ONLY the deliberate name as output.
    If there are no deliberate satisfying the request, return exactly the text '<NONE>' as the
    deliberate name, and a brief motivation explaining why the [[user request]] is out of scope.
    {}
"""

REQUEST_PARSING_PROMPT = """
    Given the user request:
    [[request]]
    "{}"
    ; and the corresponding main action to be executed to answer the request:
    [[action]]
    "{}"
    extract the (possible) inputs. The inputs should be formatted as a JSON
    and match with the action's one (see the "in:" "__in" block, if present).
    So, the inputs is a dictionary with "name" (Python variable name), "value",
    and "type" (a Python type: int, float, bool, str, list, dict, and None)
    fields that describe a single input. Note that the type None should only
    be used for variables that equal None. Lastly, output ONLY this JSON.
    {}
"""

# Used by Executor when a fallback deliberate is promoted into a new reusable
# deliberate on the same script. Asks the LLM to synthesize a stable identity
# (name / when clause / guidelines) that describes the CLASS of requests the
# new deliberate will serve, not just the current request.
FALLBACK_IDENTITY_PROMPT = """The user submitted a request that does not match any existing deliberate in the current script.
You must synthesize a NEW deliberate identity that generalises the request into a reusable deliberate.

Produce three fields:
- name: a valid CamelCase identifier (letters and digits only, starts with a letter, no spaces or punctuation).
       The name must describe the GENERAL class of requests that this deliberate will handle,
       not the specifics of the current request (e.g. "AnalyzeCpuTrend", not "AnalyzeCpuOnHostXYZOnMarch3rd").
       The name MUST NOT collide with any of the names listed below.
- when: a single concise sentence describing the class of user requests that should be routed to
       this deliberate in the future. Describe the trigger, not the implementation.
- guidelines: a brief natural-language description (no code, no bullet list) of how to solve requests
       in this class, referring to the intent only. It should read like a short operating procedure
       and will be used verbatim as the deliberate's `guidelines` block, so keep it focused and stable.

[[user_request]]
"{request}"

[[forbidden_names]]
{existing_names}
"""

# TODO: cardinality description
FRAMES_DSL = """
FRAMES DSL SYNTAX
Example frames:
```
frame Person:
    # fields
    slot name as TEXT
    slot surname as TEXT
    slot age as INT
    slot is_married as BOOL
__frame

# frame used in another frame example
frame Employee:
    slot person as Person  # <-- this field is another frame
    slot workplace as TEXT
    slot salary as FLOAT
__frame

# nested frame example
frame Picture:
	frame Size:
	    slot width as INT
	    slot height as INT
	__frame
    slot size as Size
    slot source as TEXT
__frame
```
Available slot types: [TEXT | INT | BOOL | FLOAT | STRUCT | slot_enum | frame_name]
NOTE: "slot_enum" means that a slot is an ENUM. Comply to the following syntax to 
specify the enumeration elements: slot status as ENUM ("COMPLETED", "FAILED", "IN_PROGRESS")
NOTE: "frame_name" means that a slot can be another frame, resulting in definition of 
nested frames.
NOTE: use STRUCT for both lists and dictionaries.

Strictly follow these rules:
    1. The frame must be named exactly as requested in the 'Name' field.
    2. Analyze the 'Usage Statements'. The values inside the `using [...]` may
       provide a hint about the expected frame structure.\n
    3. Output ONLY valid, parsable frame definitions.
    4. If 'Previous Frame' and an error are provided, use the previous frame as your
        baseline. Correct ONLY the logic or syntax that caused the error, preserving
        the valid parts of the frame structure.
    5. The name of the frame must not contain any white space, use camel case or snake case.
"""

GEN_FRAME_PROMPT = (
    PROLOGUE
    + """
Generate a NXS Frame based on the provided name, and custom DSL usage syntax."""
    + FRAMES_DSL
)


DO_AS_FRAMES_PROMPT = (
    PROLOGUE
    + """
Generate a NXS Frame based on the provided custom DSL usage syntax.
The generated frame must follow the description given in the `as` clause of this `do` statement and the information about the called function. 
The purpose of the frame is to give an output format for the tool/action/built-in that is being called.
Output only the frame and nothing else.
DO STATEMENT
{do_statement}
CALLABLE INFO
{callable_info}
"""
    + FRAMES_DSL
)

SCHEMA_APPLY_PROMPT = """
Map each output variable name to the most appropriate slot of frame '{frame_name}'.
Output variable names: {producing_names}
Frame '{frame_name}' slot names: {slot_names}
Return ONLY a valid Python dict literal, e.g. {{\"var_a\": \"slot_x\"}}.
Omit variables with no clear slot match.
"""

# Interpreter: semantic inclusion prompts
RIGHT_SEM_INCL_PROMPT = (
    "Semantic inclusion (a ~> b) is a form of conceptual implication or semantic "
    "specialization which is more restrictive than usual similarity "
    'between two values: means that "a" is semantically included in "b". '
    '(think about "a" and "b" as two ontologies, and check whether "a" is '
    'a subset of "b".) '
    'For example: ["dog" ~> "animal"] is true - because all dogs are'
    'animals, whereas ["animal" ~> "dog"] is false - because there'
    "are animals that are not dogs (but lions, fishes, etc). NOTE:"
    "you should think in this way to solve semantic inclusion."
)
LEFT_SEM_INCL_PROMPT = (
    "Semantic inclusion (a <~ b) is a form of conceptual implication or semantic "
    "specialization which is more restrictive than usual similarity "
    'between two values: means that "b" is semantically included in "a". '
    '(think about "a" and "b" as two ontologies, and check whether "b" is '
    'a subset of "a".) '
    'For example: ["animal" <~ "dog"] is true - because all dogs are'
    'animals, whereas ["dog" <~ "animal"] is false - because there'
    "are animals that are not dogs (but lions, fishes, etc). NOTE:"
    "you should think in this way to solve semantic inclusion."
)
SEM_INCL_TEMPLATE = (
    "Task: Evaluate the expression: [{}]. Return true or false,"
    "along with a score in 0-1 range that quantifies the degree of "
    "semantic inclusion (0: weak or none, 1: full or strong)."
)


##################################################################
CODE_SUMMARY_PROMPT = """You are an expert in the NXS programming language.

Read the following NXS code block and generate its docstring.

The docstring must be a simple plain-text description written in natural language.

It must describe, at a high level, the purpose of the block, what it does, and the main logic it follows.

Output format requirements:
Return only one plain-text paragraph.
Do not return JSON, YAML, XML, Markdown, a dictionary, an object, a list, or bullet points.
Do not include field names, labels, titles, separators, quotes, braces, brackets, or code fences.
Do not write anything before or after the docstring.
Do not explain the format.
Do not explain the code line by line.

The entire response must be only the docstring text.

NXS action block:
{action}
"""
