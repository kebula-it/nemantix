package it.kebula.plugin;

import com.intellij.psi.tree.IElementType;
import it.kebula.plugin.NemantixTypes;
import com.intellij.psi.TokenType;
import com.intellij.lexer.FlexLexer;

%%

%class NemantixLexer
%implements FlexLexer
%unicode
%function advance
%type IElementType
%eof{  return;
%eof}

// --- MACROS ---
// Define states for handling prompts intelligently
%state BLOCK_PROMPT
%state LINE_PROMPT

WHITE_SPACE=[\ \t\n\r\f]+
NEW_LINE=\r|\n|\r\n
COMMENT=#[^\r\n]*

// Identifiers (CNAME in Lark)
IDENTIFIER=[a-zA-Z_][a-zA-Z0-9_]*

// Literals
DIGIT=[0-9]
NUMBER={DIGIT}+ (\.{DIGIT}+)?
STRING=\"([^\"\\]|\\.)*\"

// NXS Specific Regexes from Lark
// NXS_PATH: /(\.\.?\/)?([A-Za-z0-9_\-]+\/)*[A-Za-z0-9_\-]+\.nxs/
PATH=(\.\.?\/)?([A-Za-z0-9_\-]+\/)*[A-Za-z0-9_\-]+\.nxs

// CARDINALITY (Simplified for Lexer: generic ranges)
// Real regex: /(\*|(0|[1-9][0-9]*)(\.\.(\*|[1-9][0-9]*))?)/
// We let '*' be handled by operators, and '1..*' by NUMBER + DOT + OPERATOR sequence generally,
// unless we want strict highlighting. Let's strictly match the complex ones.
RANGE_CARDINALITY=[0-9]+\.\.(\*|[0-9]+)

%%

<YYINITIAL> {

  // --- 1. PROMPTS (High Priority) ---

  // Start Block Prompt (>>>) -> Switch State
  ">>>"              { yybegin(BLOCK_PROMPT); return NemantixTypes.PROMPT; }

  // Start Line/Inline Prompt (>>) -> Switch State
  ">>"               { yybegin(LINE_PROMPT); return NemantixTypes.PROMPT; }

 // --- 1. PLAN QUALIFIERS (New Color) ---
  "undefined"        { return NemantixTypes.PLAN_QUALIFIER; }
  "drafted"          { return NemantixTypes.PLAN_QUALIFIER; }
  "frozen"           { return NemantixTypes.PLAN_QUALIFIER; }

  // --- 2. KEYWORDS ---

  // Structure & Definitions
  "require"          { return NemantixTypes.KEYWORD; }
  "deliberate"       { return NemantixTypes.KEYWORD; }
  "toolset"          { return NemantixTypes.KEYWORD; }
  "frame"            { return NemantixTypes.KEYWORD; }
  "slot"             { return NemantixTypes.KEYWORD; }
  "action"           { return NemantixTypes.KEYWORD; }
  "plan"             { return NemantixTypes.KEYWORD; }
  "tool"             { return NemantixTypes.KEYWORD; }
  "mandate"          { return NemantixTypes.KEYWORD; }
  "guidelines"       { return NemantixTypes.DEPRECATED_KEYWORD; }

  // Logic & Control Flow
  "if"               { return NemantixTypes.KEYWORD; }
  "elif"             { return NemantixTypes.KEYWORD; }
  "else"             { return NemantixTypes.KEYWORD; }
  "repeat"           { return NemantixTypes.KEYWORD; }
  "while"            { return NemantixTypes.KEYWORD; }
  "until"            { return NemantixTypes.KEYWORD; }
  "for"              { return NemantixTypes.KEYWORD; } // implied by generic loops
  "each"             { return NemantixTypes.KEYWORD; }
  "times"            { return NemantixTypes.KEYWORD; }
  "do"               { return NemantixTypes.KEYWORD; }
  "return"           { return NemantixTypes.KEYWORD; }
  "break"            { return NemantixTypes.KEYWORD; }
  "continue"         { return NemantixTypes.KEYWORD; }

  // Prepositions / Modifiers
  "when"             { return NemantixTypes.KEYWORD; }
  "from"             { return NemantixTypes.KEYWORD; }
  "use"              { return NemantixTypes.KEYWORD; }
  "in"               { return NemantixTypes.KEYWORD; }
  "out"              { return NemantixTypes.KEYWORD; }
  "body"             { return NemantixTypes.KEYWORD; }
  "as"               { return NemantixTypes.KEYWORD; }
  "with"               { return NemantixTypes.KEYWORD; }
  "where"            { return NemantixTypes.KEYWORD; }
  "max"              { return NemantixTypes.KEYWORD; }
  "using"            { return NemantixTypes.KEYWORD; }
  "producing"        { return NemantixTypes.KEYWORD; }
  "required"         { return NemantixTypes.INPUT_QUALIFIER; }
  "optional"         { return NemantixTypes.INPUT_QUALIFIER; }
  "default"          { return NemantixTypes.INPUT_QUALIFIER; }

  // Plan Qualifiers
  "undefined"        { return NemantixTypes.KEYWORD; }
  "drafted"          { return NemantixTypes.KEYWORD; }
  "frozen"           { return NemantixTypes.KEYWORD; }

  // Semantic Matchers
  "far"              { return NemantixTypes.KEYWORD; }
  "loose"            { return NemantixTypes.KEYWORD; }
  "about"            { return NemantixTypes.KEYWORD; }
  "close"            { return NemantixTypes.KEYWORD; }
  "strict"           { return NemantixTypes.KEYWORD; }

  // Frame Types
  "INT"              { return NemantixTypes.KEYWORD; }
  "BOOL"             { return NemantixTypes.KEYWORD; }
  "FLOAT"            { return NemantixTypes.KEYWORD; }
  "TEXT"             { return NemantixTypes.KEYWORD; }
  "ENUM"             { return NemantixTypes.KEYWORD; }
  "STRUCT"           { return NemantixTypes.KEYWORD; }

  // Values
  "true"             { return NemantixTypes.BOOLEAN; }
  "false"            { return NemantixTypes.BOOLEAN; }
  "none"             { return NemantixTypes.KEYWORD; }
  "_"                { return NemantixTypes.KEYWORD; } // Short none

  // End markers (treated as keywords for highlighting)
  "__"               { return NemantixTypes.KEYWORD; }
  "__deliberate"     { return NemantixTypes.KEYWORD; }
  "__mandate"        { return NemantixTypes.KEYWORD; }
  "__guidelines"     { return NemantixTypes.DEPRECATED_KEYWORD; }
  "__plan"           { return NemantixTypes.KEYWORD; }
  "__action"         { return NemantixTypes.KEYWORD; }
  "__body"           { return NemantixTypes.KEYWORD; }
  "__do"             { return NemantixTypes.KEYWORD; }
  "__in"             { return NemantixTypes.KEYWORD; }
  "__out"            { return NemantixTypes.KEYWORD; }
  "__repeat"         { return NemantixTypes.KEYWORD; }
  "__if"             { return NemantixTypes.KEYWORD; }
  "__toolset"        { return NemantixTypes.KEYWORD; }
  "__use"            { return NemantixTypes.KEYWORD; }
  "__frame"          { return NemantixTypes.KEYWORD; }

  // --- 3. OPERATORS ---
  "->"               { return NemantixTypes.OPERATOR; }
  "~>"               { return NemantixTypes.OPERATOR; }
  "<~"               { return NemantixTypes.OPERATOR; }
  "=="               { return NemantixTypes.OPERATOR; }
  "!="               { return NemantixTypes.OPERATOR; }
  "<="               { return NemantixTypes.OPERATOR; }
  ">="               { return NemantixTypes.OPERATOR; }
  "??"               { return NemantixTypes.OPERATOR; }
  "||"               { return NemantixTypes.OPERATOR; }
  "&&"               { return NemantixTypes.OPERATOR; }
  "^^"               { return NemantixTypes.OPERATOR; }
  "~"                { return NemantixTypes.OPERATOR; }
  "+"                { return NemantixTypes.OPERATOR; }
  "-"                { return NemantixTypes.OPERATOR; }
  "*"                { return NemantixTypes.OPERATOR; }
  "/"                { return NemantixTypes.OPERATOR; }
  "%"                { return NemantixTypes.OPERATOR; }
  "^"                { return NemantixTypes.OPERATOR; }
  "="                { return NemantixTypes.OPERATOR; }
  "!"                { return NemantixTypes.OPERATOR; }
  "<"                { return NemantixTypes.OPERATOR; }
  ">"                { return NemantixTypes.OPERATOR; }
  "|"                { return NemantixTypes.OPERATOR; }

  // --- 4. PUNCTUATION ---
  "{"                { return NemantixTypes.LBRACE; }
  "}"                { return NemantixTypes.RBRACE; }
  "["                { return NemantixTypes.LBRACKET; }
  "]"                { return NemantixTypes.RBRACKET; }
  "("                { return NemantixTypes.LPAREN; }
  ")"                { return NemantixTypes.RPAREN; }
  ","                { return NemantixTypes.COMMA; }
  ":"                { return NemantixTypes.COLON; }
  "."                { return NemantixTypes.DOT; }
  "@" [a-zA-Z_][a-zA-Z0-9_]* ("." [a-zA-Z_][a-zA-Z0-9_]*)* { return NemantixTypes.META; }

  // --- 5. COMPLEX TOKENS ---

  {PATH}             { return NemantixTypes.PATH; }
  {RANGE_CARDINALITY} { return NemantixTypes.NUMBER; }
  {NUMBER}           { return NemantixTypes.NUMBER; }
  {IDENTIFIER}       { return NemantixTypes.IDENTIFIER; }
  {STRING}           { return NemantixTypes.STRING; }
  {COMMENT}          { return NemantixTypes.COMMENT; }
  {WHITE_SPACE}      { return TokenType.WHITE_SPACE; }

  [^]                { return TokenType.BAD_CHARACTER; }
}

  // --- STATE HANDLING ---
  <BLOCK_PROMPT> {
    // End of block
    "<<<"              { yybegin(YYINITIAL); return NemantixTypes.PROMPT; }

    // Consume any char that isn't the start of the closer
    [^<]+              { return NemantixTypes.PROMPT; }
    "<"                { return NemantixTypes.PROMPT; }
  }

  <LINE_PROMPT> {
    // Case A: Inline closer found -> End prompt, go back to code
    "<<"               { yybegin(YYINITIAL); return NemantixTypes.PROMPT; }

    // Case B: Newline found -> End prompt, return whitespace, go back to code
    {NEW_LINE}         { yybegin(YYINITIAL); return TokenType.WHITE_SPACE; }

    // Content: Anything that isn't a newline or the start of a closer
    [^<\r\n]+          { return NemantixTypes.PROMPT; }
    "<"                { return NemantixTypes.PROMPT; }
}