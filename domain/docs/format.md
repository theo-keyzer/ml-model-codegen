# File Formats Documentation

Schema and data file formats for the generator system

---

## Core Formats

### line_based_format

Fixed-field line-based format for unit, act, and artifact files with elements in schema-defined sequence.

#### Format Characteristics

Each line represents one component instance. Fields appear in fixed sequence matching Element declarations in schema. Whitespace (spaces or tabs) separates fields. Only one text field allowed per component, must be last field. Optional fields use `.` as placeholder filler.

#### No Nesting or Tree Structure

Format is flat with no indentation-based nesting. Child components appear after parent in file order. Parent-child relationship determined by component type not position. Order of components in file does not matter for resolution.

#### Parent-Child Flow

Child automatically goes to known parent that appears before it. Parent field name is automatically the parent component's name. Top-level nodes can have optional parent field to reference another node via reference.

---

### hocon_format

HOCON-style format for user data files with no nesting and additive key semantics.

#### Format Characteristics

Key-value pairs with `key = value` syntax. No nested blocks or hierarchies. Keys are not overwritten, values accumulate. Brace syntax creates components: `ComponentName { field1 = value1, field2 = value2 }`

#### No Nesting Structure

Format is flat with no block nesting inside other blocks. Each component block stands alone. Parent-child relationships defined by order and explicit parent field.

#### Parent Field Validation

User must specify `parent = ParentName` in child components. System validates that parent appears before child in file. System validates parent-child relationship matches schema. Order does not matter except parent must precede child.

---

## Schema Files (Unit Format)

### unit_file_format

Schema definition files using line-based format to define component types and element fields.

**Prerequisites:** line_based_format

#### File Extension

Schema files use `.unit` extension (e.g., `artifact.unit`, `tsu.unit`, `gen.unit`).

#### Component Declaration

```
Comp ComponentName parent ParentRef SearchType
```

Defines new component type. ParentRef is parent component name or `.` for root. SearchType is Find, FindIn, or `.`

#### Element Declaration

```
Element fieldname datatype targetcomp validation documentation
```

Defines field within component. Fields appear in this sequence in data files. Last element can be text type capturing to end of line.

#### Example Unit File

```
Comp Artifact parent . Find
    Element name     key  .        + Unique identifier
    Element category word .        * Optional category
    Element doc      text .        * Documentation

Comp Link parent Artifact FindIn
    Element concept  ref  Artifact + Target artifact
    Element relation ref  Artifact + Relationship type
```

---

## Actor Files (Act Format)

### actor_file_format

Code generation templates using line-based format for processing data and producing output.

**Prerequisites:** line_based_format

#### File Extension

Actor files use `.act` extension (e.g., `g_struct.act`, `view.act`, `p-bit.act`).

#### Actor Declaration

```
Actor actorname ComponentName attr = value
```

Defines actor with optional match condition. Multiple actors with same name form match group.

#### Command Syntax

Commands appear indented under actor. Common commands: `C` (output line), `All` (iterate), `Its` (navigate), `Add` (to collection), `Break` (exit), `Out` (control output).

#### Example Actor File

```
Actor main
    All Model generate_model

Actor generate_model Model
C # Generated code for ${model}
Its Layer generate_layer

Actor generate_layer Layer
C // Layer: ${layer}
```

---

## Artifact Files (Artifact Format)

### artifact_file_format

Documentation files using line-based format for AI-friendly knowledge management.

**Prerequisites:** line_based_format

#### File Extension

Artifact files use `.artifact` extension (e.g., `facts.artifact`, `docs.artifact`).

#### Artifact Declaration

```
Artifact name category topic_type
```

Category and topic_type are optional words. Doc field not used in definitions.

#### Overview Lines

```
O This is overview text.
```

Multiple O lines allowed. No parent prefix needed, automatically children of preceding Artifact.

#### Link Declarations

```
Link target_artifact relation_type
```

Both fields validated as references to existing artifacts. No parent prefix needed.

#### Section Structure

```
Section name level heading_text
```

Level is numeric (2, 3, 4) for heading depth. Optional for short artifacts.

#### Document Lines

```
D Content text with markdown.
```

Must be children of Section. Multiple D lines allowed per section.

#### Example Artifact File

```
Artifact my_concept structure overview
O Brief description of concept.

Link foundation prerequisite
Link related_topic related

Section details 2 Detailed Information
D First paragraph of details.
D Second paragraph with **markdown**.
```

---

## Data Files (HOCON Format)

### data_file_format

User data files using HOCON-style format for model specifications and configurations.

**Prerequisites:** hocon_format

#### File Extension

Data files use `.net` or domain-specific extensions (e.g., `p-bit.net`, `model.net`).

#### Component Syntax

```
ComponentName {
  field1 = value1
  field2 = value2
  parent = ParentName
}
```

Brace pairs create component instances. Parent field required for child components.

#### Field Values

Single-line values for most fields. String values can be quoted or unquoted. Reference fields contain target component name. Multi-line blocks use triple-quote or triple-tilde delimiters.

#### Multi-Line Blocks

Triple single-quotes `'''` delimit code blocks. Triple tildes `~~~` delimit markdown blocks. Start delimiter appears at end of key line. End delimiter on its own line. Delimiters can nest opposite type: `'''` blocks can contain `~~~` and vice versa.

#### Parent Ordering Requirement

Parent component must appear before child in file. System validates parent exists and matches schema. Child components specify `parent = ParentComponentName`.

#### Example Data File

```
Model {
  model = MyNetwork
  type = energy_based
  desc = "My model description"
}

Kernel {
  kernel = my_kernel
  hardware = gpu_device
  body =
'''
void kernel_func(float* data) {
  // Kernel implementation
  // Can contain ~~~ markdown ~~~
}
'''
}

Layer {
  layer = sampling_layer
  type = sampling
  parent = MyNetwork
  desc =
~~~
This layer performs **sampling** with:
- Feature 1
- Feature 2

Can contain ''' code blocks '''
~~~
}
```

---

## Format Comparison

### format_comparison

Comparison of line-based and HOCON formats showing their different use cases.

#### Line-Based Format Uses

- **Schema definition** (unit files) - Define component types and validation rules
- **Code generation** (act files) - Template-driven output generation
- **Documentation** (artifact files) - Structured knowledge management with validation

#### Line-Based Advantages

- Compact fixed-field syntax
- Visual alignment with tabs
- Single text field captures long descriptions
- Implicit parent-child from component type sequence

#### HOCON Format Uses

- **User data** (net files) - Model specifications, configurations, domain data
- **Human-friendly input** - Named fields, readable structure
- **Validation against schema** - Explicit parent fields verify correctness

#### HOCON Advantages

- Named fields are self-documenting
- Field order doesn't matter
- Explicit parent validation
- No column alignment required
- Easier for humans to write

---

## Field Order and Validation

### field_ordering

Fixed element sequence in line-based format matches schema declaration order.

**Prerequisites:** line_based_format

#### Schema Defines Sequence

Element declarations in unit file define field order. Data files must provide fields in this exact sequence. Skipped optional fields use `.` placeholder.

#### Text Field Restriction

Only one text type element allowed per component. Text element must be last in sequence. Text field captures all remaining content to end of line.

#### Example

```
# Schema
Element name     word . + Name
Element category word . * Category  
Element doc      text . * Documentation

# Data file
Artifact my_artifact structure This is documentation text
Artifact other      .         No documentation provided
```

---

### parent_child_mechanics

Automatic parent-child relationship resolution in both formats.

#### Implicit Parents (Line-Based)

Child component type determines its parent type from schema. Child instance attaches to most recent parent instance in file. No explicit parent field needed in data. Parent field in component defines reference to another node.

#### Explicit Parents (HOCON)

Child components must specify `parent = ParentName` field. System looks for parent component instance with matching name. Validates parent type matches schema definition. Reports error if parent not found or wrong type.

#### Top-Level Parent References

Top-level Find components can have parent element in schema. Parent element is typically a reference (ref) to another component. In data file, provides the referenced component name. Creates link between top-level nodes without nesting.

#### Example

```
# Line-based (unit file defines parent type)
Comp Section parent Artifact FindIn

# Line-based (data file - implicit parent)
Artifact my_artifact structure overview
Section intro 2 Introduction

# HOCON (explicit parent validation)
Section {
  section = intro
  level = 2
  parent = my_artifact
}
```

---

## Parsing and Loading

### parsing_rules

Rules for parsing both line-based and HOCON formats during loading.

#### Line-Based Parsing

Split line on whitespace for word fields. First token is component name. Subsequent tokens map to element sequence. Last text field captures remaining content. Empty lines and dash lines ignored.

#### HOCON Parsing

Parse brace blocks as component instances. Extract component type from block header. Parse key-value pairs inside braces. Field names map to element definitions. Order of fields within block does not matter.

#### Reference Resolution

Both formats load all components first. Then resolve all references in multiple passes. Validate parent-child relationships. Report unresolved references as errors.

---

### multi_line_blocks

Multi-line block syntax in HOCON format for embedding code and markdown content.

**Prerequisites:** hocon_format

#### Block Delimiters

Triple single-quotes `'''` mark code block boundaries. Triple tildes `~~~` mark markdown block boundaries. Opening delimiter appears at end of field assignment line. Closing delimiter appears alone on its own line.

#### Nesting Capability

Code blocks (`'''`) can contain markdown delimiters (`~~~`) as literal text. Markdown blocks (`~~~`) can contain code delimiters (`'''`) as literal text. This allows documenting one format within the other without escaping.

#### Use Cases

Code blocks for kernel implementations, function bodies, templates. Markdown blocks for rich documentation with formatting. Use opposite delimiter type when content contains the other delimiter.

#### Example

```
Kernel {
  kernel = example_kernel
  body =
'''
void my_kernel() {
  // Code here
  /* Can mention ~~~ safely */
}
'''
  documentation =
~~~
This kernel does **X**.

Example usage:
```
call_kernel();
```

Can mention ''' delimiters safely.
~~~
}
```

---

### filler_placeholders

Placeholder values for optional fields in line-based format.

**Prerequisites:** line_based_format

#### Dot Placeholder

Single dot `.` represents omitted optional field. Required for positional parsing. Maintains field alignment and sequence. Common in optional category, type, or validation fields.

#### When To Use Fillers

Use `.` when element validation is `*` (optional). Skip value for fields you don't need. Maintains correct field position for subsequent fields. Not needed in HOCON format (named fields).

#### Example

```
# With optional fields filled
Artifact my_artifact structure overview Full documentation

# With optional fields as placeholders  
Artifact other      .         .        No category or type

# HOCON - no placeholders needed
Artifact {
  name = my_artifact
  doc = "Full documentation"
}
```

---

## Format Design Rationale

### format_design_rationale

Design decisions behind two-format approach balancing different needs.

#### Schema Needs Precision

Line-based format enforces strict structure. Fixed field order prevents ambiguity. Compact syntax for frequently edited files. Visual alignment improves readability for schemas.

#### Users Need Flexibility

HOCON format is self-documenting with named fields. Field order flexibility reduces errors. Explicit parent validation catches mistakes. More familiar to users from other config systems.

#### Separation of Concerns

Schemas define structure strictly (line-based). User data provides flexibility (HOCON). Generator developers work in line-based format. End users work in HOCON format. Both validate to same schema ensuring consistency.

---

## Quick Reference

### Format Summary Table

| Aspect | Line-Based | HOCON |
|--------|------------|-------|
| **File Types** | .unit, .act, .artifact | .net, domain-specific |
| **Field Order** | Fixed (schema sequence) | Flexible (named) |
| **Optional Fields** | `.` placeholder required | Omit entirely |
| **Parent-Child** | Implicit from type | Explicit with validation |
| **Text Fields** | One per component (last) | Any number of fields |
| **Use Case** | Schema/template authors | End users/data providers |
| **Nesting** | None (flat) | None (flat) |
| **Validation** | Reference resolution | Reference + parent validation |

### When to Use Each Format

**Use Line-Based Format when:**
- Defining schemas (unit files)
- Writing code generation templates (act files)
- Creating documentation with validation (artifact files)
- You need compact, aligned syntax
- You're a framework developer

**Use HOCON Format when:**
- Providing user data (model specs, configs)
- You want self-documenting field names
- Field order flexibility is important
- You need explicit parent validation
- You're an end user or domain expert
