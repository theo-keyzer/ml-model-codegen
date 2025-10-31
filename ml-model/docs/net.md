Introduction to .net File Structure
This document explains the hierarchical structure of .net model specification files and what each component type is used for.
File Structure Overview
A .net file defines your neural network model in a tree structure:

Model                          (Root - your entire model)
├── Tensors                    (Data containers)
├── Layers                     (Logical groupings)
│   ├── Operations            (Computations)
│   │   └── Arguments         (Input/output connections)
│   └── BlockInstances        (Reusable components)
├── Blocks                     (Reusable templates)
│   └── BlockParams           (Parameters for templates)
├── ControlFlow               (Conditional execution)
│   ├── Conditions            (When to execute)
│   └── Branches              (What to execute)
├── ContinuousLayers          (Neural ODEs)
└── Configs                   (Execution settings)
    └── Schedules             (Execution order)
```

## Core Components

### Model (Root)

The top-level container for your entire neural network.
```
Model {
  model = TransformerMoE
  desc = "Transformer with Mixture of Experts"
}
```

**Purpose**: Names and describes your model architecture.

**Key Properties**:
- `model` - Unique identifier
- `desc` - Human-readable description
- `type` - Optional: static, dynamic, searchable

---

### Tensor

Defines data containers (inputs, outputs, weights, activations).
```
Tensor {
  tensor = input_tokens
  parent = TransformerMoE
  shape = [1, 512]
  layout = nc
  dtype = int32
  desc = "Input token IDs"
}
```

**Purpose**: Specifies the shape, type, and memory layout of data.

**Key Properties**:
- `tensor` - Unique name
- `parent` - Which model it belongs to
- `shape` - Dimensions (e.g., `[batch, seq_len, hidden]`)
- `shape_type` - `static` or `dynamic`
- `dyn_dims` - Which dimensions can vary (e.g., `1` for sequence length)
- `layout` - Memory layout (`nc`, `ncd`, `nhwc`, etc.)
- `dtype` - Data type (`fp32`, `int32`, etc.)

**Common Layouts**:
- `nc` - (batch, channels) for 2D
- `ncd` - (batch, channels, depth) for 3D
- `nhwc` - (batch, height, width, channels) for images
- `oi` - (output, input) for weight matrices

---

### Layer

Groups related operations into logical units.
```
Layer {
  layer = mha_layer
  parent = TransformerMoE
  type = attention
  desc = "Multi-head attention"
}
```

**Purpose**: Organizes operations hierarchically (like a module or block).

**Key Properties**:
- `layer` - Unique identifier
- `parent` - Which model it belongs to
- `type` - Layer category (attention, dense, recurrent, etc.)
- `desc` - Description of what this layer does

**Common Types**:
- `attention` - Attention mechanisms
- `dense` - Fully connected layers
- `embedding` - Token/positional embeddings
- `recurrent` - RNN/LSTM/GRU
- `graph` - Graph neural networks
- `continuous` - Neural ODEs
- `mixture_of_experts` - MoE layers

---

### Op (Operation)

Defines a single computation within a layer.
```
Op {
  op = self_attention
  parent = mha_layer
  kernel = attention_cuda
  kernel_op = attention_dispatch
  desc = "Causal multi-head self-attention"
}
```

**Purpose**: Specifies what computation to perform and which kernel to use.

**Key Properties**:
- `op` - Operation name
- `parent` - Which layer it belongs to
- `kernel` - Reference to kernel implementation
- `kernel_op` - Dispatch/launch configuration
- `op_rule` - Operation type (for specialized ops)

---

### Specialized Operation Types

#### AttentionOp

For attention mechanisms with query, key, value.
```
AttentionOp {
  attn_op = self_attention
  parent = self_attention
  attn_type = multi_head
  num_heads = 12
  head_dim = 64
  causal = true
}
```

**Used for**: Self-attention, cross-attention, multi-head attention.

#### GraphOp

For graph neural network operations.
```
GraphOp {
  graph_op = graph_attention
  parent = graph_attention
  gnn_type = gat
  aggregation = attention
  num_layers = 2
}
```

**Used for**: GCN, GAT, GraphSAGE, message passing.

#### StatefulOp

For recurrent operations that maintain hidden state.
```
StatefulOp {
  state_op = lstm_cell
  parent = lstm_cell
  state_type = lstm
  stateful = true
}
```

**Used for**: LSTM, GRU, custom RNNs.

#### ODEOp

For neural ordinary differential equations.
```
ODEOp {
  ode_op = ode_integrate
  parent = ode_integrate
  integration = dopri5
  solver = dopri5
  adjoint = true
}
```

**Used for**: Neural ODEs, continuous-depth networks.

#### ExpertRoutingOp

For mixture of experts routing.
```
ExpertRoutingOp {
  expert_op = moe_router
  parent = weighted_combine
  num_experts = 8
  top_k = 2
  experts = expert_ffn
  load_balance = token_choice
}
```

**Used for**: Sparse MoE, conditional computation.

---

### Arg (Argument)

Connects operations to tensors (inputs, outputs, parameters).
```
Arg {
  arg = query
  parent = self_attention
  role = query
  tensor = query_tensor
}
```

**Purpose**: Specifies which tensors flow into/out of operations.

**Key Properties**:
- `arg` - Argument name
- `parent` - Which operation it belongs to
- `role` - Purpose of this argument
- `tensor` - Which tensor to use

**Common Roles**:
- `input` - Standard input
- `output` - Standard output
- `param` - Learnable parameter
- `query`, `key`, `value` - For attention
- `hidden_state`, `cell_state` - For recurrent ops
- `node_features`, `adjacency` - For graph ops
- `gate_tensor`, `expert_output` - For MoE

---

## Reusable Components

### Block

Template for reusable layer patterns.
```
Block {
  block = attention_block
  block_type = attention
  model = TransformerMoE
  layers = mha_layer
  parameters = num_heads_param
  desc = "Multi-head attention block"
}
```

**Purpose**: Define once, instantiate many times with different parameters.

### BlockParam

Parameters that can be customized per instance.
```
BlockParam {
  param = num_heads_param
  parent = attention_block
  param_type = int
  default = 8
  desc = "Number of attention heads"
}
```

### BlockInstance

Concrete use of a block with specific parameters.
```
BlockInstance {
  instance = encoder_block
  parent = transformer_layer
  block = attention_block
  repeat = 6
  desc = "Stack 6 encoder blocks"
}
```

**Purpose**: Create N repeated blocks (like stacking transformer layers).

### ParamValue

Override default parameter values for an instance.
```
ParamValue {
  param = num_heads_param
  parent = encoder_block
  value = 12
  desc = "Use 12 heads instead of default 8"
}
```

---

## Control Flow

### ControlFlow

Define conditional or dynamic execution paths.
```
ControlFlow {
  control = adaptive_routing
  parent = TransformerMoE
  type = conditional
  desc = "Adaptive expert routing based on input"
}
```

**Purpose**: Enable dynamic computation graphs.

**Types**:
- `conditional` - If/else branches
- `loop` - Repeated execution
- `switch` - Multi-way branches

### Condition

When to execute a branch.
```
Condition {
  condition = should_route
  parent = adaptive_routing
  predicate = threshold
  input = gate_logits
  threshold = 0.5
}
```

**Purpose**: Evaluate whether to execute a code path.

**Predicates**:
- `threshold` - Compare value to threshold
- `tensor_value` - Based on tensor contents
- `expert_router` - Based on routing decisions

### Branch

What to execute when condition is met.
```
Branch {
  branch = expert_path
  parent = adaptive_routing
  branch_id = 0
  layers = expert_layer
  desc = "Execute expert network"
}
```

**Purpose**: Define alternative execution paths.

---

## Advanced Features

### ContinuousLayer

Neural ODE with continuous depth.
```
ContinuousLayer {
  cont_layer = ode_layer
  parent = TransformerMoE
  solver = dopri5
  time_steps = 10
  t_start = 0
  t_end = 1
  adaptive = true
  dynamics = ode_dynamics
  tolerance = 0.001
}
```

**Purpose**: Continuous-depth networks where layer depth is learned.

### GraphTensor

Special tensors for graph-structured data.
```
GraphTensor {
  graph_tensor = input_graph_nodes
  parent = TransformerMoE
  graph_type = node_features
  num_nodes = 1000
  feature_dim = 768
  sparse_format = csr
  dtype = fp32
}
```

**Purpose**: Represent nodes, edges, and adjacency matrices.

**Graph Types**:
- `node_features` - Node feature vectors
- `edge_features` - Edge attributes
- `adjacency` - Graph connectivity

**Sparse Formats**:
- `csr` - Compressed sparse row
- `coo` - Coordinate format
- `edge_list` - List of edges

---

## Execution Configuration

### Config

Specifies target device and optimization settings.
```
Config {
  config = inference_adaptive
  parent = TransformerMoE
  target = gpu_a100
  batch = 1
  opt_flags = [fused_ops, fp16_mixed, sparse_attention]
  desc = "Adaptive inference with dynamic routing"
}
```

**Purpose**: Define how and where to run the model.

**Key Properties**:
- `target` - Hardware (gpu_a100, cpu_x86, etc.)
- `batch` - Batch size
- `opt_flags` - Optimizations to apply

### Schedule

Defines execution order of operations.
```
Schedule {
  seq = 1
  parent = inference_adaptive
  layer = input_embed
  op = token_embed
  desc = "Token embedding"
}
```

**Purpose**: Specify the sequence of operations to execute.

**Key Properties**:
- `seq` - Order number (1, 2, 3...)
- `layer` - Which layer to execute
- `op` - Specific operation (optional, all if omitted)
- `control` - Control flow instead of layer

---

## Tree Navigation Rules

### Parent-Child Relationships
```
Model
  ├── Tensor (parent = Model)
  └── Layer (parent = Model)
      └── Op (parent = Layer)
          └── Arg (parent = Op)
```

Every child must reference its parent with `parent = ParentName`.

### Reference Relationships

Components can reference other components:
```
Arg {
  tensor = query_tensor  // References a Tensor
}

Op {
  kernel = attention_cuda  // References a Kernel
}

ExpertRoutingOp {
  experts = expert_ffn  // References a Layer
}
```

### Navigation in Properties

Access related components:
- `parent.property` - Parent component's property
- `tensor.shape` - Referenced tensor's shape
- `kernel.backend` - Referenced kernel's backend

---

## Typical Model Structure

A complete model follows this pattern:
```
1. Model declaration
2. All Tensors (inputs, weights, activations, outputs)
3. All Layers
   - Each Layer contains Ops
   - Each Op contains Args
4. Blocks (if using reusable components)
5. ControlFlow (if using dynamic execution)
6. ContinuousLayers (if using Neural ODEs)
7. Configs
   - Each Config contains Schedules

Best Practices

Name consistently: Use descriptive names like attention_output, not tensor_42
Group logically: Keep related operations in the same layer
Document: Use desc fields for complex components
Start simple: Build up complexity incrementally
Reuse blocks: For repeated patterns (transformer layers, residual blocks)
Dynamic shapes: Use shape_type = dynamic for variable-length inputs

Next Steps

Study moe.net to see a complete model structure
Start with a simple feedforward network
Add complexity incrementally (attention → MoE → control flow)
Use blocks for repeated patterns
Test each layer before adding the next

The tree structure makes it easy to understand model architecture at a glance and enables powerful code generation through hierarchical traversal.
