# Working on uni-assistant

## Purpose

This repository is a learning project, not just a calculator. The owner wants
to understand, modify, and debug the mathematics by reading the implementation.
Prefer a clear algorithm with visible steps over a shorter opaque solution.
Use the current code and README as the source of truth for the public API.

## Keeping these instructions current

- When the owner explicitly adds or changes an ongoing project preference,
  update this AGENTS.md as part of the same task so future agents follow it.
- Replace outdated or conflicting guidance rather than appending contradictory
  rules. Keep updates focused and preserve unrelated instructions.
- Distinguish ongoing preferences from one-task exceptions; do not turn a
  temporary request or an inferred preference into a permanent rule.

## Before changing code

- Read the relevant README section, implementation, and callers. Check the
  working-tree and staged diffs; preserve changes already present.
- Prefer codebase-memory-mcp tools for discovery: `search_graph`, `trace_path`,
  `get_code_snippet`, then `query_graph` or `get_architecture` as needed.
  If tools are unavailable, the project is not indexed, or results are
  insufficient, use `rg`. Use `rg` for literals, configuration, and prose too.
- Keep the change within the requested scope. An algorithm fix, API redesign,
  and structural refactor are different tasks; do not silently combine them.
- Use installed Caveman and Ponytail skills when available. Keep user-facing
  updates concise; write code and documentation in clear, normal English.
  Respect requests to leave those modes.
- Use `rtk` for shell commands when available; `rtk proxy <command>` preserves
  raw output. Otherwise run the underlying command directly.

## Where code belongs

| File | Responsibility |
| --- | --- |
| `linear_algebra/matrix.py` | Public `Matrix` API, storage, construction, operators, elementary row operations |
| `linear_algebra/_matrix_algebra.py` | Arithmetic, determinant/cofactors, shared scalar operations |
| `linear_algebra/_row_reduction.py` | RREF and pivots; solving, inverse, rank, and spaces built from them |
| `linear_algebra/_matrix_vectors.py` | Vector products, coordinates, span, orthogonality, projection |
| `linear_algebra/_matrix_spectral.py` | Characteristic polynomial, eigenvalues, eigenspaces, diagonalization, similarity |
| `linear_algebra/polynomial.py` | SymPy polynomial interface and reduction of exact root expressions |
| `linear_algebra/_matrix_types.py` | Shared type aliases |
| `linear_algebra/config.py` | Existing numerical tolerance |
| `linear_algebra/linear_transformation.py` | Linear maps built on our `Matrix` API |
| `sample_main.py` | Small executable learning example and assertions |
| `README.md` | Usage, mathematical flow, reading order, and API migration notes |

The separate `Vector`, `Vec3d`, custom complex-number, and logic modules have
their own APIs. Do not rewrite them as a side effect of matrix work.

## Library boundary

- Keep matrix algorithms handwritten: multiplication, determinant, elimination,
  solving, inverse, spaces, eigenspaces, and diagonalization belong in this repo.
- SymPy may handle polynomials, roots, exact scalar expressions, and polynomial
  fields used for arithmetic on algebraic numbers. Do not delegate matrix
  algorithms to SymPy matrices, `DomainMatrix`, NumPy, or SciPy.
- Reuse native SymPy `Poly`; do not recreate a custom polynomial algebra system.
- Prefer existing helpers, the standard library, and existing dependencies.
  Add a dependency only when the task needs it, and declare it in
  `requirements.txt`.

## Adding a feature

1. Define its mathematical input, output, and failure cases. Decide whether it
   accepts rectangular matrices, complex values, or symbolic expressions.
2. Find the existing operation that supplies most of the work. For example,
   an eigenspace is a null space; do not write another elimination routine.
3. Implement the algorithm in the relevant topic module. Expose a small,
   explicitly defined method in `Matrix` when it belongs in the public API.
4. Follow the conventions below. Add a new module only for a distinct topic
   that no existing module reasonably owns, not for each new method.
5. Update the relevant README section following the documentation guidance
   below, and verify a mathematical identity through the public API.

## README and usage documentation

- Update README.md in the same task whenever a change affects how users create
  inputs, call an operation, interpret results, or handle errors and limitations.
  Keep examples consistent with the current API and docstrings.
- Write for someone new to the project: they should be able to install it and
  use its main features by reading the README without inspecting implementation
  files or knowing earlier conversations.
- Organize usage by topic with descriptive headings: setup and a quick start,
  matrix construction and arithmetic, symbolic inputs, systems and spaces,
  vector coordinates and projection, polynomials and equations, eigenvalues
  and diagonalization, linear transformations, and logic. Add examples to the
  relevant category rather than appending unrelated snippets at the end.
- Explain when to use each operation, its expected inputs and output shape,
  and important restrictions. Include small runnable examples with imports,
  defined inputs, and expected results; introduce basic usage before advanced
  details. Distinguish project methods from direct SymPy usage.
- Keep setup and usage easy to find. Put implementation reading guides and
  migration notes after the introductory usage material, and update navigation
  links when sections move. Verify changed examples and links before handoff.

## API and numerical conventions

- Use descriptive `snake_case` names and useful argument/return annotations.
  Keep definitions explicit so editor completion and go-to-definition work.
  Prefer familiar mathematical notation or abbreviations for public Matrix
  methods: `T()`, `inv()`, `adj()`, `proj()`, `diag()`, `rot90()`, `P()`, and
  `D()`. The owner also prefers `coords`, `ortho_coords`, `is_ortho`, `col_space`,
  `from_cols`, `swap_cols`, `without_col`, `alg_mult`, `geom_mult`, `cof_matrix`,
  and `can_diag` over their former long names. Use the short names without
  keeping long-name aliases. Explain their full mathematical meaning in
  docstrings; `can_diag()` specifically means diagonalizable, not already diagonal.
- Use `@` for matrix multiplication and `*` for scalar multiplication.
- Matrices contain rectangular row lists. Vectors and solution vectors are
  columns; space methods return lists of column vectors. This also applies to
  the vector representation returned by `row_space()`.
- Return new matrices without mutating or aliasing inputs. Helpers can use
  `matrix._new(rows)` or existing constructors. Direct edits to `data` are the
  caller's explicit mutation path.
- Preserve exact integer/Fraction calculations. Do not introduce floats by
  dividing integers directly where the existing `divide()` helper is needed.
- Reuse `is_zero()`, `simplify_scalar()`, and `config.TOLERANCE` where appropriate.
  Exact values use exact zero tests; floating calculations use the configured
  tolerance. Never round intermediate eigenvalues or silently cast roots to floats.
- Reuse `reduce_rows()` and its pivot information. Keep solving, inverse,
  rank, and spaces consistent with that routine.
- Validate shapes and unsupported inputs with clear exceptions. Do not return
  `None` for an invalid operation or use assertions for public input validation.
- `Matrix([])` represents 0 by 0; other zero-dimension shapes are unsupported.
  Keep unsupported cases explicit instead of returning a misleading shape.
- Preserve current names and return contracts unless the task authorizes API
  changes. For an authorized change, update callers, examples, annotations,
  and the README migration table together.

## Refactoring and keeping code readable

- For a structural refactor, preserve behavior, return shapes, exceptions,
  exactness, and mutation semantics. Make mathematical changes explicit.
- Trace all callers before changing a shared helper. Fix the common cause
  rather than adding guards to individual callers.
- Prefer focused functions, ordinary loops, and visible intermediate values.
  A reader should be able to connect each step to the mathematical method.
- Keep runtime imports acyclic. Helpers use `TYPE_CHECKING` for `Matrix` type
  references; do not add per-function imports to hide a dependency cycle.
- Avoid registries, factories, mixins, dynamic method injection, generic plugin
  systems, and configuration added only for hypothetical future needs.
- Add or update clear docstrings when adding or changing functions and methods,
  especially public APIs. Explain what the operation is for, its inputs, return
  value and shape, and important restrictions or failure cases. Keep simple
  operations brief; include a small mathematical example when meaning is unclear.
- Explain distinctions between related operations. For example, `coords()`
  finds coefficients that reconstruct a vector in an independent basis;
  `ortho_coords()` finds factors for nonzero orthogonal directions
  (projection coefficients when the target is outside their span); `proj()`
  returns the projected vector rather than its scalar factor.
- Keep docstrings consistent with behavior when code changes. Retain clear
  structure and names; use brief inline comments for non-obvious mathematical
  steps and put longer tutorials in the README.
- Document real algorithmic limits. For a deliberate shortcut, a brief
  `ponytail:` comment should name the limit and what would replace it.
  Do not hide a different algorithm behind an automatic fallback.

## Verification and handoff

- For code changes, run the existing example and checks relevant to the changed
  behavior. Useful identities include `A @ solution == b`, `A @ v == lambda*v`,
  `A @ P == P @ D`, and `A @ A.inv() == I`; use `is_close()` as appropriate.
- Cover applicable edge cases: singular/rectangular systems, dependent vectors,
  repeated eigenvalues, exact fractions, floats, irrational or complex roots,
  and input immutability. Algebraic-root changes need a `CRootOf` case too.
- Keep a small runnable assertion in the existing example for substantive new
  behavior when useful. Use temporary scripts or in-memory checks for broader
  validation. Do not add repository test files or a test framework unless asked.
- Respect explicit instructions not to run tests. Documentation-only changes
  need content/link and whitespace checks, not numerical regression runs.

```bash
rtk proxy python sample_main.py
rtk proxy python -m compileall -q linear_algebra sample_main.py
rtk proxy git diff --check
```

Review the final diff. Report what changed, what actually ran, and any remaining
limits. Do not claim correctness from compilation alone. Do not stage, commit,
or revert unrelated work; commit only when requested and after validation.
