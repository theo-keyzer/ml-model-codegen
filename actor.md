# Actor System - Complete Documentation

## Overview

The Actor System is a template-based code generation engine that processes hierarchical data structures through pattern matching and conditional execution. It combines declarative templates with imperative control flow for flexible code generation.

## Core Concepts

### Execution Model
- **Actors**: Named template units that execute when their conditions match
- **Windows**: Execution contexts that maintain state during processing
- **Navigation**: Path-based data access through component hierarchies
- **Conditions**: Pattern matching with safe null handling and logical chaining

## Basic Syntax

### Actor Definition
```
Actor <name> <context> <condition>
<commands>
```

### Command Types
- **`C <text>`**: Output line with newline
- **`Cs <text>`**: Output snippet without newline
- **`All <what> <actor> <args>`**: Execute actor for all matching components
- **`Its <what> <actor> <args>`**: Execute actor for child components
- **`This <path> <actor> <args>`**: Execute actor for data at path
- **`Du <actor> <args>`**: Add logic to current actor (shared context)
- **`Break <what>`**: Control flow termination
- **`Add <path> <data>`**: Data manipulation
- **`Out <mode>`**: Output control

## Variable System

### Path Resolution
- **`${tensor}`**: Simple variable substitution
- **`${tensor:l}`**: With transformation (lowercase)
- **`${._lno}`**: System variables (line numbers)
- **`${parent.child}`**: Hierarchy navigation
- **`${.sibling.attr}`**: Window stack lookup

### Special Variables
- **`._prev_cnt`**: Previous loop count (child iterations + 1)
- **`._cnt`**: Current iteration count
- **`._depth`**: Call stack depth
- **`._key`**: Current map key
- **`._type`**: Current data type

## Condition System

### Operators
- **`=`**: Equality
- **`!=`**: Inequality
- **`?=`**: Null-safe equality (false if null/error)
- **`&=`**: Logical AND with previous condition
- **`in`**: List membership
- **`has`**: List contains
- **`regex`**: Pattern matching

### Condition Patterns
```
# Single condition
Actor process Op type = conv

# Safe attribute access
Actor process Op platform ?= cuda

# Condition chaining
Actor process Op type = conv
Actor process Op stride &= 1

# Default case (no conditions)
Actor process Op
```

## Control Flow

### Loop Commands
- **`All`**: Process all matching components
- **`Its`**: Process child components
- **`This`**: Process path-based data

### Break Types
- **`Break actor`**: Stop current actor chain
- **`Break loop`**: Stop current loop
- **`Break cmds`**: Stop command list
- **`Break exit`**: Terminate script

### Context Sharing
- **`Du`**: Shares variables, counters, and data with called actor
- Preserves `Cnt`, `PrevCnt`, `DataKey`, `DataType`, `DataKeys`

## Navigation Patterns

### Hierarchy Navigation
```
# Parent access
Its parent.Arg process_args

# Sibling access  
Its parent.Child process_siblings

# Cross-hierarchy
Its parent.parent.Config apply_settings

# Link access from Arg
Its tensor tensor_settings

# Reverse access from Tensor
Its Arg_tensor arg_settings
```

### Window Stack Lookup
```
# Find actor in call stack
Actor specialize Op type = ${.template_op.type}

# Configuration propagation
Actor generate Op platform = ${.config.platform}
```

## Collection System

### Data Storage
- **Global `Collect` map**: Cross-actor data sharing
- **Path-based access**: Dot notation for nested data
- **Type support**: Maps, lists, and nested structures

### Collection Commands
```
# Add to collection
Add _.tensors:${name} ${data}

# Check existence
Add.check.break _.tensors:${name} ${data}

# Map operations
Add.map _:config
Add     _.config:platform cuda
```

## Error Handling

### Error Types
- **Load errors**: Schema and parsing issues
- **Runtime errors**: Execution and navigation failures
- **Condition errors**: Safe null handling with `?=` operator

### Error Recovery
- Continues execution after non-fatal errors
- Line number tracking for debugging
- Conditional safety prevents crash on missing data

## Execution Context

### Window State
```go
type WinT struct {
    Name     string      // Actor name
    Cnt      int         // Iteration counter
    PrevCnt  int         // Previous loop count
    Dat      interface{} // Current data
    DataKey  string      // Current map key
    DataType string      // Current data type
    DataKeys []string    // Available keys
    // Control flags...
}
```

### Counter Semantics
- **`Cnt`**: Current iteration index (0-based)
- **`PrevCnt`**: Child loop iterations + 1
- **Empty loop**: `PrevCnt = 0` when no children processed

## Best Practices

### Actor Design
- Use descriptive names for different behaviors
- Place default cases last in actor chains
- Use `Break` for exclusive matching
- Leverage `Du` for modular logic

### Condition Safety
- Prefer `?=` for optional attributes
- Chain conditions with `&=` for complex logic
- Use collection checks to prevent duplicates

### Navigation
- Use relative paths for maintainability
- Leverage window stack for configuration
- Prefer explicit over implicit navigation

## Common Patterns

### Template Processing
```
Actor generate_file Model
C # Header
All Component generate_component
C # Footer
```

### Conditional Generation
```
Actor process Op type = conv
C # Conv-specific code
Break actor

Actor process Op type = pool
C # Pool-specific code  
Break actor

Actor process Op
C # Default implementation
```

### Validation Chain
```
Actor validate Model
Du check_structure
Du check_compatibility
Du check_performance

Actor check_structure Model layers > 0
C # Structure valid
```

This system enables sophisticated code generation through composable templates, safe pattern matching, and flexible data navigation.
