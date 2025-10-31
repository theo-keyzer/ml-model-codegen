Introduction to Neural Network DSL
This document provides a quick introduction to building neural network models using our domain-specific language (DSL). The DSL separates what you want to build (your model specification in .net files) from how to generate code (actor scripts in .act files).
Core Concepts
Three File Types

.unit files - Schema definitions that define the structure of components (like a database schema)
.net files - Your model specifications using the schema (what you want to build)
.act files - Code generation scripts (how to transform your model into executable code)

Separation of Concerns

Domain Knowledge (Kernels, Optimizations) - Reusable implementation details
Model Specification (Layers, Operations, Tensors) - Your neural network architecture
Code Generation (Actors) - Automated transformation to target platforms

Building a Simple Model
1. Define Your Model
Start by declaring a model and its components:

Model {
  model = MyTransformer
  desc = "A simple transformer model"
}
```

### 2. Define Tensors

Specify your data structures with shapes and types:
```
Tensor {
  tensor = input_tokens
  parent = MyTransformer
  shape = [1, 512]
  layout = nc
  dtype = int32
  desc = "Input token IDs"
}
```

### 3. Create Layers

Organize operations into logical layers:
```
Layer {
  layer = attention_layer
  parent = MyTransformer
  type = attention
  desc = "Multi-head attention"
}
```

### 4. Add Operations

Define computations within layers:
```
Op {
  op = self_attention
  parent = attention_layer
  kernel = attention_cuda
  desc = "Self-attention computation"
}
```

### 5. Connect with Arguments

Link operations to tensors:
```
Arg {
  arg = query
  parent = self_attention
  role = input
  tensor = query_tensor
}
```

## Advanced Features

### Dynamic Shapes

Support variable-length sequences:
```
Tensor {
  tensor = dynamic_input
  shape = [1, 512, 768]
  shape_type = dynamic
  dyn_dims = 1  // sequence dimension is dynamic
}
```

### Mixture of Experts

Define expert routing:
```
ExpertRoutingOp {
  expert_op = moe_router
  num_experts = 8
  top_k = 2
  load_balance = token_choice
}
```

### Control Flow

Add conditional execution:
```
ControlFlow {
  control = adaptive_routing
  type = conditional
}

Condition {
  condition = should_route
  predicate = threshold
  input = gate_logits
  threshold = 0.5
}
```

### Reusable Blocks

Create parameterizable components:
```
Block {
  block = transformer_block
  block_type = sequential
}

BlockInstance {
  instance = encoder_block
  block = transformer_block
  repeat = 6  // stack 6 blocks
}
```

## Code Generation with Actors

Actors traverse your model specification and generate target code:
```
Actor generate_model_files Model
C // Generated CUDA code
C #include <cuda_runtime.h>
Its Tensor declare_tensor
All Op generate_op_implementation
```

### Actor Patterns

- **`All`** - Iterate over all matching components
- **`Its`** - Iterate over children of current component
- **`C`** - Output a line of code
- **`${variable}`** - Access component properties
- **`Break`** - Stop and use next matching actor

## Configuration and Scheduling

Define how your model executes:
```
Config {
  config = inference_config
  target = gpu_a100
  batch = 1
  opt_flags = [fused_ops, fp16_mixed]
}

Schedule {
  seq = 1
  layer = embedding_layer
  op = token_embed
}

Next Steps

Study Examples - Review moe.net for a complete transformer with MoE
Understand Schemas - Read net5.unit to see available components
Learn Actors - Examine moe2.act to understand code generation patterns
Start Simple - Begin with a basic feedforward network before tackling complex architectures

Key Principles

Declarative - Describe what you want, not how to compute it
Composable - Build complex models from reusable components
Target-Agnostic - Same model can generate CUDA, Metal, or other backends
Type-Safe - Schema validation catches errors early
Extensible - Add new operation types without changing the core system

The DSL enables you to focus on model architecture while automating the tedious work of kernel implementation, memory management, and platform-specific optimizations.

