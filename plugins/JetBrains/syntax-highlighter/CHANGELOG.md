# Changelog

## [1.1.4]
### Bugfix
- Introduced `mandate` as a replacement for the `guidelines`
keyword - `guidelines` is still kept but deprecated for backward compatibility.
- Improved the code formatter.
- Fixed how comments are handled inside a `toolset` block.

## [1.1.3]
- General code pasting improvements.

### Bugfix
- Deprecation warning.
- Fix struct indentation in code pasting.
- Python language injection occurs when there is actual Python code in `toolset` blocks.

## [1.1.2]
### Bugfix
- Python code within `toolset` blocks caused erroneous indentation on snippet pasting.
- Indentation of one-line copy and pasting is now preserved.

## [1.1.1]
### Bugfix
- Missing `BOOL` keyword in `slot` types within `frame` definitions.

## [1.1.0]
### Added
- Automatic code formatting on paste.

## [1.0.9]
### Bugfix
- Missing highlight of `none` and `_` in expressions and struct definitions.

## [1.0.8]
### Bugfix
- Removed keyword highlight in variables definition and assignment.
- Keywords `required`, `optional` and `default` are now correctly highlighted in `in` blocks.

## [1.0.7]
### Bugfix
- Keyword used as fields within Struct definition are not anymore highlighted.

## [1.0.6]
### Added
- Highlight of Python code within `toolset` blocks.

## [1.0.5]
### Added
- Live templates (i.e., code snippets) for `deliberate`, `action`, `toolset` and `frame`.

## [1.0.4]
### Added
- Complete highlight of "intentables" definitions. 
  E.g., now `@intent.goal` is fully highlighted - previously only `@` was.

## [1.0.3]
### Added
- Highlight of `require` keyword.
- Changed highlight color for micro-prompts. 

### Removed
- Highlight of `include` keyword due to syntax change.
- `editable` keyword plan qualifier.

## [1.0.2]
### Added
- Folding of `guidelines`, `toolset` and `frame`.
**NOTE:** folding is enabled when the block is terminated with `__block_name` instead of `__`.
- Now, when folding `deliberate`, `action`, `toolset`, and `frame`, the corresponding name
is shown instead of `...`.


## [1.0.1]
### Added
- Support for `.nxv` file extension.**
