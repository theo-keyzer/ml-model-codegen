Introduction to Actor Scripts
This document explains how to write actor scripts (.act files) that generate code from your neural network specifications. Actors traverse your model components and output target code (CUDA, C++, etc.).
Core Concepts
What Are Actors?
Actors are code generation templates that:

Match specific component types from your .net model
Access component properties using ${property} syntax
Output code lines using C prefix
Navigate relationships between components

Basic Structure

Actor actor_name ComponentType
C // Generated code line
C // Another line with ${property} interpolation
```

## Actor Syntax

### 1. Code Output

Lines starting with `C` generate code:
```
Actor generate_function Op
C void op_${op}_forward() {
C     printf("Executing ${op}\\n");
C }
```

### 2. Property Access

Use `${property}` to access component attributes:
```
Actor declare_tensor Tensor
C extern float* ${tensor}; // ${shape} ${dtype}
```

### 3. Conditional Properties

Use `::` for defaults when property might be missing:
```
C int num_steps = ${time_steps::10};  // defaults to 10
C bool adaptive = ${adaptive::false}; // defaults to false
```

### 4. Property Modifiers

Transform property values:

- `${property:l}` - Convert to lowercase
- `${property:u}` - Convert to uppercase
- `${property:0}` - Get first element of list
- `${property:-1}` - Get last element of list
```
C void ${op:l}_kernel() {  // lowercase op name
```

## Navigation Patterns

### Its - Iterate Over Children

Process child components of the current element:
```
Actor generate_layer Layer
C // Layer: ${layer}
Its Op generate_op_call
```

### All - Iterate Over All Matches

Process all components of a type in the model:
```
Actor generate_model Model
C // Model: ${model}
All Tensor declare_tensor
All Op generate_op_implementation
```

### Upward Navigation

Access parent components:
```
Actor generate_op Op
C // Operation: ${op}
C // Parent layer: ${parent.layer}
```

## Control Flow

### Break Statement

Stop matching actor case switch:
```
Actor generate_op_implementation Op kernel_op ??
C // Fallback version when no KernelOp exists
Break

Actor generate_op_implementation Op
C // Use KernelOp version
```

The `??` means "only if property does not exists".

### Conditional Actors

Use property checks to select specific actors:
```
Actor attention_get_mask Arg role = mask
Cs mask,   // 'Cs' continues previous line
Break

Actor attention_get_mask Arg
Cs NULL,  // default case
```

## Common Patterns

### 1. Declaration Pattern

Generate forward declarations first, implementations later:
```
Actor declare_op_forward Op
C void op_${parent.layer:l}_${op:l}_forward(void* stream);

Actor generate_op_implementation Op
C void op_${parent.layer:l}_${op:l}_forward(void* stream) {
C     // implementation
C }
```

### 2. Argument Passing Pattern

Collect arguments from child components:
```
Actor generate_kernel_launch Op
C kernel<<<grid, block>>>(
Cs        
Its Arg generate_kernel_args
C );

Actor generate_kernel_args Arg
Cs ${.1.,} ${tensor.tensor:l}  // comma-separated list
```

The `${.1.,}` adds commas between items (but not before first).

### 3. Nested Navigation Pattern

Navigate through multiple levels:
```
Actor generate_attention AttentionOp
Its parent.Arg attention_get_qkv_tensors

Actor attention_get_qkv_tensors Arg role = query
C float* query = ${tensor.tensor:l};
```

### 4. Conditional Generation Pattern

Generate different code based on properties:
```
Actor generate_solver ODEOp
C // Solver: ${integration}
Its parent.Arg ode_declare_tensors
C
C if (adaptive) {
C     printf("Using adaptive stepping\\n");
C }
```

## Special Syntax

### Add.map

Create lookup tables for iteration:
```
Add.map _:tensor
All Model generate_model_files
```

### Add.break

Skip duplicates:
```
Actor declare_tensor Tensor
Add.break _.tensor:${tensor:l}
C extern float* ${tensor:l};
```

### Cs - Continue Line

Continue previous line without newline:
```
C kernel<<<grid, block>>>(
Cs arg1,    // continues previous line
Cs arg2     // continues previous line  
C );        // new line
```

## Property Reference Syntax

### ${.property}

Access current component's property:
```
C int value = ${.threshold};
```

### ${parent.property}

Access parent component's property:
```
C // Parent layer: ${parent.layer}
```

### ${component.property}

Access referenced component's property:
```
C float* data = ${tensor.tensor:l};
```

### ${property.subproperty}

Navigate through references:
```
C // Shape: ${tensor.shape:0}  // first dimension
```

## Practical Example

Here's a complete actor for generating operation implementations:
```
Actor generate_op_implementation Op
C
C void op_${parent.layer:l}_${op:l}_forward(void* stream) {
C     // Operation: ${op}
C     // Kernel: ${kernel}
C     // From: ${._lno}
C
C     // Setup launch configuration
C     int block_size = ${kernel_op.block_size};
C     dim3 block(block_size);
C     dim3 grid(${kernel_op.grid_calc});
C
C     // Kernel launch
C     ${kernel:l}_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
Cs        
Its Arg generate_kernel_args
C     );
C     cudaStreamSynchronize((cudaStream_t)stream);
C }

Actor generate_kernel_args Arg
Cs ${.1.,} ${tensor.tensor:l}
```

## Best Practices

1. **Use descriptive actor names** that indicate what they generate
2. **Add comments in generated code** with metadata (line numbers, descriptions)
3. **Handle missing properties** with `::default` syntax
4. **Use Break wisely** to create fallback patterns
5. **Group related actors** together with comment headers
6. **Test incrementally** - start simple and add complexity

## Common Patterns by Use Case

### Generating Declarations
```
Actor declare_X X
C extern type ${name};
```

### Generating Implementations
```
Actor generate_X_impl X
C void ${name}_function() {
Its Child generate_child_call
C }
```

### Conditional Code Generation
```
Actor generate_with_feature X feature ??
C // Feature enabled
Break

Actor generate_without_feature X
C // Feature disabled
```

### List Processing
```
Actor process_list Parent
Its Child process_item

Actor process_item Child
Cs ${.1.,} ${value}

Next Steps

Study moe2.act - See real-world patterns in action
Start small - Begin with simple declaration actors
Test frequently - Generate code and check output
Build libraries - Create reusable actor patterns
Debug with metadata - Use ${._lno} to track source lines

The actor system is designed to be intuitive - if you can write the target code manually, you can template it with actors.

