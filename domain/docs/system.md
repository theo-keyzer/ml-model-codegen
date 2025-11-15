# Artifact Documentation System

AI-friendly granular knowledge management

---

## Relation Types

### prerequisite
Must understand this concept first before proceeding.

### related
Provides additional context or complementary information.

### extension
Builds upon or extends the linked concept.

### example
Provides concrete demonstration or code sample.

### contrast
Shows alternative approach or different solution.

---

## Categories

### structure
Data structure definitions, schemas, and type systems describing how data is organized.

### workflow
Processes and sequential operations that accomplish tasks.

### pattern
Design patterns and reusable solutions to common problems.

### principle
Fundamental rules and guidelines governing system behavior.

---

## Topic Types

### overview
High-level summary introducing concepts without deep technical detail.

### detail
Implementation-level information with precise technical specifications.

### example
Concrete code or data demonstrating the concept in practice.

### reference
Quick lookup information for experienced users.

---

## System Architecture

Multi-stage template-based code generation system separating schema definition, data content, and generation logic.

**Links:** bootstrap_process (related), runtime_data_model (extension)

### Core Generator

Core generator in Go reads unit schemas and generates loader code in target language.

Command: `go run gen/*.go actor_file schema1.unit,schema2.unit >output`

### Application Generator

Generated loader reads definition files, validates against schema, and uses actor files to drive code generation.

### Bootstrap Benefits

Generator regenerates its own loader when schemas change. New applications require no host language coding. Schema changes propagate automatically through regeneration.

---

## bootstrap_process

Two-stage bootstrap where the generator generates itself.

**Prerequisites:** system_architecture

### Stage 1: Core Generator

Hand-written Go code in `gen/` directory generates `structs.go` and `run.go` from unit schemas.

### Stage 2: Application Generator

Generated loader processes definition files using actor files to produce output.

---

## runtime_data_model

Generated code creates data structures holding parsed nodes with polymorphic Kp interface.

**Prerequisites:** system_architecture

### Kp Interface

All component types implement Kp interface with DoIts, GetVar, and GetLineNo methods enabling polymorphic handling.

### Node Structure

Each component generates KpCompName struct with Me, LineNo, Comp, Flags, Names map, parent pointer, reference pointers, and child arrays.

---

## Schema Definition

### unit_file_structure

Unit files define schemas using component-element model as meta-schema describing definition file structure.

**Prerequisites:** component_declaration, element_definition

---

### component_declaration

Components define node types appearing in definition files with name, parent relationship, and searchability.

**Prerequisites:** unit_file_structure

#### Comp Statement

```
Comp ComponentName parent ParentRef SearchType
```

ComponentName is unique identifier. ParentRef references parent or `.` for root. SearchType is Find, FindIn, or `.`

#### Search Types

Find marks top-level components searchable by name globally. FindIn marks nested components searchable within parent context.

---

### element_definition

Elements define fields within components specifying field names, data types, target components, and validation rules.

**Prerequisites:** component_declaration

#### Element Statement

```
Element fieldname datatype targetcomp validation documentation
```

Datatype specifies parsing and storage. Validation controls error handling (`.`, `+`, `1`, `*`).

#### Field Types

- **word** - Single whitespace-delimited token
- **text** - All remaining text to end of line
- **number/tree** - Numeric depth for tree navigation
- **ref** - Links to top-level Find components
- **link** - Links to sibling FindIn components

---

### reference_types

Reference types create links between nodes enabling graph structures. Each reference has name field and namep pointer.

**Prerequisites:** element_definition

#### Direct References

- **ref** links to top-level Find components by name
- **link** links to sibling FindIn components under same parent

#### Derived References

- **ref_copy** copies reference ID from preceding ref/link field in target component
- **ref_child** looks up child component within previously referenced component

Both depend on previous ref/link being resolved first.

---

### reference_resolution

References resolve in multiple passes after data loading until stable or errors remain.

**Prerequisites:** reference_types

#### Resolution Algorithm

First pass resolves ref and link references. Subsequent passes resolve ref_copy and ref_child. Process continues until error count is zero or unchanged. Typical schemas resolve in 2-4 passes.

---

## Runtime Execution

### actor_execution

Actors are function-like processors with match conditions that process nodes and generate output.

**Prerequisites:** runtime_data_model

#### Actor Groups

Multiple actors with same name form group. System tries each sequentially. First actor with matching condition executes. If no match, group call fails silently.

#### Match Conditions

```
Actor actorname ComponentName attr = value
```

Component field filters types. Attribute and operator specify comparison. Supports =, !=, in, not-in, has, regex operators.

---

### command_types

Commands within actors perform iteration, code generation, and control flow.

**Prerequisites:** actor_execution

#### Iteration Commands

- **All** - Calls actor for each component of type
- **Its** - Navigates via references or children
- **This** - Iterates collection items
- **Du** - Calls actor with current node

#### Code Generation

- **C** - Outputs line with variable substitution
- **Cs** - Outputs without newline
- **Out** - Controls output timing (delay, normal, off, on)
- **In** - Redirects output to variable

#### Control Flow

**Break** - Exits actors, loops, or command lists with actor, loop, cmds, or exit options.

---

### variable_system

Variables use dollar-brace syntax for substitution with path resolution and modifiers.

**Prerequisites:** actor_execution

#### Variable Syntax

- `${name}` substitutes value
- `${name:c}` capitalizes
- `${name:u}` uppercase
- `${name:l}` lowercase
- `${?name}` optional without error

#### Variable Scope

- `${fieldname}` accesses current node field
- `${_.name}` accesses collection
- `${.actorname.var}` accesses ancestor variable
- `${parent}` navigates to parent

#### Special Variables

- `${.-}` loop counter (zero-based)
- `${.+}` loop counter (one-based)
- `${kMe}` node ID
- `${kComp}` component name
- `${N}` command-line argument N

---

### collection_storage

Collections provide dynamic storage during generation for accumulating data and cross-actor communication.

**Related:** command_types

#### Collection Types

Collections store ordered lists or key-value maps. System infers type from first Add operation. Collections persist across all actors.

#### Add Operations

- `Add path value` adds to collection
- `Add .map path:key` creates map
- `Add .list path:key` creates list
- Options: Clear, Break, Check, No-add

#### Collection Access

- `This` iterates collection calling actor for each item
- `${_.name}` accesses string values
- `${_.name.field}` accesses node fields

---

## Advanced Patterns

### common_patterns

Design patterns for common generation tasks using collections and actor groups.

**Prerequisites:** actor_execution
**Related:** collection_storage

#### Filter Pattern

Use collections with check and break options to track processed nodes and avoid duplicates.

#### Multi-Pass Generation

First pass collects information into collections. Subsequent passes use collections to generate output. `Out delay` defers output until next actor.

#### Separator Pattern

Use loop counter `${.-}` to test for first item. `${.-.,}` generates comma only when counter is nonzero.

---

### collection_iteration

Dot notation controls collection iteration granularity at container or item level.

**Extension of:** collection_storage

#### Iteration Levels

- `This collection actor` processes entire container
- `This collection. actor` processes each item individually

Critical for proper collection processing.

#### List Iteration

Container-level receives entire list. Item-level receives each element. Use `${.-}` for zero-based index, `${.+}` for one-based.

#### Map Iteration

Container-level receives entire map. Item-level receives key-value pairs. Use `${._key}` for key, `${._value}` for value.

---

### path_resolution

Path chaining enables deep property access through nodes and collections.

**Extension of:** variable_system

#### Path Chaining

- `${_.collection.nested.property}` chains segments
- `${.parent.grandparent.var}` navigates through actors
- Each segment navigates through nodes or collections

#### Collection Operations

- `${.:join}` joins with commas
- `${.:join: }` joins with spaces
- `${.:count}` counts items
- `${.:keys}` gets keys list
- `${.:values}` gets values list

---

### recursive_output

Using Out delay for clean hierarchical output in recursive algorithms.

**Extension of:** command_types

#### The Problem

Without output control, recursive algorithms produce interleaved output mixing parent and child results.

#### Out Delay Solution

`Out delay` buffers all subsequent output until actor completion, then releases in proper order maintaining hierarchy.

---

## Artifact System

### artifact_system

Documentation and knowledge management format designed for both human readability and AI consumption.

**Prerequisites:** artifact_components

#### System Goals

Enable precise AI context retrieval. Provide structural validation for AI-generated content. Support multiple views from single source. Scale to hundreds of artifacts without overhead.

#### Key Features

Granular units of 10-30 lines perfect for AI context windows. Typed relationships create validated knowledge graph. Optional metadata only when adding value. Self-documenting system structure.

---

### artifact_components

Artifact is fundamental knowledge unit with metadata, overview, and optional sections.

**Prerequisites:** artifact_system

#### Artifact Declaration

```
Artifact name category topic_type
```

Name is unique identifier. Category and topic_type are optional references to classification artifacts.

#### Overview Content

```
O This is the overview text.
```

Multiple O lines concatenate. Should be 1-3 sentences capturing essence.

#### Section Structure

```
Section name level Heading Text
```

Optional structure for longer content. Level indicates heading depth (2, 3, 4).

---

### link_relationships

Links create typed relationships forming validated knowledge graph enabling navigation and ensuring structural integrity.

**Prerequisites:** artifact_components

#### Link Syntax

```
Link target_artifact relation_type
```

Target and relation must both exist and are validated by loader.

#### Purpose of Links

Create learning paths through prerequisite chains. Enable contextual navigation. Build concept hierarchies. Connect theory to practice. Show alternatives.

---

### validation_rules

Artifact format enforces validation through unit schema and loader with strict link validation.

**Prerequisites:** artifact_system, link_relationships

#### Loader Validation

Artifact names must be unique. Link targets must reference existing artifacts. Relation types must be valid. Sections must be children of artifacts.

#### Not Validated

Category names can be any word. Topic type names can be any word. Section content accepts any text. This flexibility allows evolution without schema changes.

---

### optional_metadata

Category and topic_type are optional metadata. Create defining artifacts only when they add value beyond their names.

**Related:** artifact_components

#### The Value Test

Before creating category or type artifact ask: Does this definition change how someone would use or understand the system? Create if concept is complex, ambiguous, or requires standards. Skip if name is obvious or just restates the word.

#### Metadata as Labels

Think of categories and types as labels for filtering and organization, not strict schemas. Only Links require validation because they create structural relationships.

---

### design_philosophy

Artifact format balances AI-friendliness for consumption with structural rigor for generation.

**Prerequisites:** artifact_system

#### Two Goals

AI-friendly consumption needs small focused units, easy retrieval, minimal dependencies. AI-correct generation needs structural constraints, validated relationships, type safety.

#### The Balance

Strict where it matters: graph structure via Links, component hierarchy, reference integrity. Flexible where it helps: classification via categories, documentation types, content format.

---

### ai_context_optimization

Artifact format optimizes for AI context windows by providing granular focused knowledge units for precise retrieval.

**Prerequisites:** artifact_system

#### The Context Problem

Traditional documentation has monolithic structure with large chapters, mixed abstraction levels, unclear dependencies. For AI queries this wastes token budget and degrades response quality.

#### The Artifact Solution

Each artifact is 10-30 lines with one clear purpose. Explicit dependencies via Links. Easy to identify relevant units. Enables surgical context injection instead of sending entire chapters.

#### Progressive Context Loading

AI requests additional context as needed. Each round adds 15-25 lines not 100+ lines. Builds minimal relevant context automatically through Link traversal.

---

## Development Workflow

### development_workflow

Creating custom generator follows structured process from design through iterative refinement.

**Prerequisites:** component_declaration, actor_execution

#### Design Phase

Identify entities and relationships in problem domain. Sketch data hierarchy and cross-references. Plan output format generation.

#### Schema Development

Create unit file defining components and elements. Start with simple top-level components. Add child components and references with appropriate types.

#### Actor Development

Start with simple actors using C commands. Test navigation commands. Build up complexity incrementally. Use collections to track state.

---

### example_applications

System enables diverse generation tasks from documentation to database schemas to web applications.

**Prerequisites:** development_workflow

#### Documentation Generator

Define Concept, Topic components for structured docs. Use number field for hierarchical nesting. Actors generate HTML with proper heading levels.

#### Database Schema Generator

Define Type, Attr, Where components. Use ref references for foreign keys. Actors generate CREATE TABLE statements, ORM classes, views.

#### Loader Generator

The g_struct.act and g_run.act actors generate the loader itself from gen.unit and act.unit, demonstrating complete self-hosting capability.
