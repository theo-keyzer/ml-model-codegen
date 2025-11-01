# Net5 Schema Understanding and Modification Guide

## Project Context

This guide is for working with the **existing Net5 schema** in the `ml-model-codegen` project. The system consists of:

- **`net5.unit`**: Schema definition (components, elements, relationships)
- **`res50_net.txt`**: Example model definitions (ResNet-50)
- **`net3_act.txt`**: Code generation templates
- **Generated code**: CUDA kernels, validation code, documentation

**Note**: For creating entirely new schema systems from scratch, see the separate `domain-codegen` project documentation.

## Understanding the Net5 Schema

### Schema File Structure

The `net5.unit` file defines the component hierarchy and validation rules:

```
Comp ComponentName parent ParentComponent FindType
----------------------------------------------------------------
* Documentation description
----------------------------------------------------------------

	Element field_name type target_comp check * Description
		Opt value . * Option documentation
```

### Core Components Hierarchy

#### Domain Knowledge (Expert HOW)
- **Domain**: Knowledge repository for ML operations
- **Kernel**: Hardware-specific implementations  
- **KernelOp**: Operation dispatch knowledge
- **KernelParam**: Kernel parameter definitions
- **Optimization**: Target-specific optimization strategies

#### User Specification (User WHAT) 
- **Model**: ML model definition
- **Layer**: Neural network layers
- **Op**: Base operations (inherited by specific ops)
- **Tensor**: Data tensors with static/dynamic shapes
- **Config**: Inference configurations
- **Schedule**: Execution order

#### Specialized Operations
- **AttentionOp**: Self/cross/multi-head attention
- **GraphOp**: Graph neural network operations  
- **StatefulOp**: RNNs, GRUs, LSTMs
- **ODEOp**: Neural differential equations
- **ExpertRoutingOp**: Mixture of experts
- **And 15+ other operation types...**

### Element Types and Validation

#### Reference Types
- `ref`: Direct reference to top-level component
- `link`: Reference to sibling component  
- `type_of`: Reference by child type (polymorphic)
- `ref_child`: Lookup child within referenced component
- `up_copy`: Navigate upward from parent context
- `ref_copy`: Copy reference from previous element

#### Value Types
- `key`: Unique identifier (required for Find/FindIn)
- `word`: Single word value
- `text`: Multi-line string
- `number`: Numeric value
- `tree`: Numeric with hierarchy support

#### Validation Levels (`check` field)
- `.`: Default validation (error if not found, unless ".")
- `+`: Required (error if missing)
- `1`: Exactly one match required
- `*`: Optional (no error if missing)

### Reference Chain System

Certain element types form lookup chains using previous elements as context:

```yaml
# Chain example: Model → Layer → Op
Element model    up_copy    Model    # Navigate up to Model
Element layer    ref_child  Layer    # Find Layer in Model  
Element op       ref_child  Op       # Find Op in Layer
```

**Chain Rules:**
- Elements must be ordered (context flows left to right)
- `*_copy` elements are system-generated (never in input files)
- Broken chains cause resolution errors

## Definition Files Format

### Basic Syntax
Definition files use flat syntax with implicit hierarchy:

```yaml
Model {
  model = MyModel
  desc = "Model description"
}

Layer {
  layer = conv1
  type = conv
  desc = "Convolution layer"
  # parent = MyModel (optional validation)
}

Op {
  op = conv_op
  kernel = matmul_cuda
  # parent = conv1 (optional validation)  
}
```

### Parent Resolution Rules

1. **Implicit Resolution**: Children use most recent compatible parent
2. **Tree Order**: Files must follow depth-first hierarchy order
3. **Explicit Validation**: `parent` field validates against implicit resolution
4. **Error Conditions**:
   - Child before any parent → Error
   - Explicit parent ≠ Implicit parent → Error  
   - Invalid parent type → Error

### Example: ResNet-50 Structure

```yaml
Model { model = ResNet50 }

# conv1 layer and children
Layer { layer = conv1 }
Op { op = im2col_conv }
Arg { arg = input, tensor = input_img }
Arg { arg = output, tensor = conv1_output }

# res_block1 layer and children  
Layer { layer = res_block1 }
Op { op = matmul }
Arg { arg = input, tensor = bn_output }
```

## Modifying the Schema

### Adding New Components

1. **Define the component** in `net5.unit`:

```yaml
Comp NewOp parent Op FindIn
----------------------------------------------------------------
* New operation type for specialized hardware
----------------------------------------------------------------

	Element new_op    key   . + Operation identifier
	Element hardware  ref   Hardware + Target hardware
	Element feature   word  . * Special feature flag
	Element desc      text  . * Description
```

2. **Update related components** if needed:

```yaml
# Add to OpTypeRule
OpTypeRule {
  op_type = new_op
  category = specialized
  desc = "New operation type"
}
```

3. **Regenerate the parser** (see Regeneration Process below)

### Adding New Fields to Existing Components

```yaml
Comp Layer parent Model FindIn
----------------------------------------------------------------
* Enhanced layer specification
----------------------------------------------------------------

	Element layer      key  . + Layer identifier
	Element type       word . + Layer type
	# New field:
	Element precision  word . * Precision (fp32, fp16, int8)
	Element desc       text . * Description
```

### Creating New Operation Types

1. **Define the operation**:

```yaml
Comp NewSpecializedOp parent Op FindIn
----------------------------------------------------------------
* Operation for new research domain
----------------------------------------------------------------

	Element spec_op    key   . + Operation name
	Element domain     word  . + Research domain
	Element params     text  . * Special parameters
```

2. **Add to OpTypeRule**:

```yaml
OpTypeRule {
  op_type = new_specialized
  category = research
  desc = "For new research domain"
}
```

3. **Update ArgRoleRule** if new argument roles needed

## Regeneration Process

Only need to regenerate if Comp types are added
or reference elements are changed.
The other fields are dynamic.

After modifying `net5.unit`:

```bash
# 1. Regenerate parser from domain-codegen project
Use domain-codegen project
See bld/srch.sh for hocon input file format.

Only the run.go and struct.go is generated.
The rest stay the same.
Errors in the unit file are generate, compile or load errors.

```

## Common Modification Scenarios

### Adding Hardware Support

1. Add new `TargetRule` for the hardware
2. Create specialized `Kernel` components
3. Add hardware-specific `Optimization` strategies
4. Update actor templates for new code generation

### Supporting New Neural Network Types

1. Add new operation types (`Comp NewOp parent Op FindIn`)
2. Define appropriate argument roles in `ArgRoleRule`
3. Create domain knowledge (`Kernel`, `KernelOp`)
4. Extend code generation templates

### Enhancing Tensor System

1. Add new tensor types or layouts
2. Extend `DtypeRule` for new data types
3. Add shape inference rules
4. Update memory layout templates

## Best Practices for Modifications

### Backward Compatibility
- Add new fields as optional (`check = *`)
- Provide default values where possible
- Use new component types rather than modifying existing ones
- Test with existing model definitions

### Validation Design
- Use `+` for critical fields that must be resolvable
- Use `1` when exactly one match is required
- Use `*` for optional features
- Provide clear error messages through documentation

### Hierarchy Design
- Keep tree depth reasonable (2-4 levels ideal)
- Use `FindIn` for components that belong to parents
- Use `Find` for globally accessible components
- Consider performance of reference resolution

## Testing Modifications

### Validation Tests
```bash
# Test schema parsing
./bin/net5_parse --validate-only res50_net.txt

# Test code generation
./bin/net5_parse res50_net.txt net3_act.txt > /dev/null

# Test new component definitions
./bin/net5_parse test_new_components.txt net3_act.txt
```

### Example Test File
Create `test_modification.def`:

```yaml
# Test new components
Model { model = TestModel }

# Test new fields
Layer { 
  layer = test_layer 
  type = conv
  precision = fp16  # New field
}

# Test new operation types
NewSpecializedOp {
  spec_op = test_special
  domain = neuromorphic
}
```

## Troubleshooting

### Common Errors After Modification

**Parser Generation Failures:**
- Syntax errors in `net5.unit`
- Circular dependencies in parent references
- Invalid element types or options

**Definition File Errors:**
- Missing required fields (check `+` validation)
- Broken reference chains
- Invalid parent-child relationships

**Code Generation Issues:**
- Missing actor templates for new components
- Incorrect context references in templates
- Type mismatches in generated code

### Debugging Reference Resolution
- Use explicit `parent` fields for validation
- Check component order in definition files
- Verify `ref_*` chain ordering
- Test with minimal examples

## Next Steps

- Review existing `res50_net.txt` for usage patterns
- Study `net3_act.txt` for code generation templates
- Test modifications with small example files
- Consult `domain-codegen` docs for advanced schema design

This guide covers understanding and modifying the existing Net5 schema. For fundamental changes to the schema system itself, refer to the `domain-codegen` project documentation.
