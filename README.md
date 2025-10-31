ml-model-codegen

💡 Project Overview

ml-model-codegen is a cutting-edge Machine Learning Model Code Generation framework designed to translate high-level model definitions into highly-optimized, target-specific source code (e.g., CUDA, C++).

This project allows researchers and engineers to define complex neural network architectures, like those with Mixture-of-Experts (MoE) and dynamic graph operations, using a structured domain-specific language (DSL). The framework then utilizes a rule-based actor system and domain knowledge repository to generate production-ready, high-performance inference code.

✨ Key Features

    Advanced Model Support: Seamlessly define and generate code for complex architectures, including Transformer-based models with Mixture-of-Experts (MoE) and adaptive/conditional execution.

    High-Performance Generation: Generates optimized low-level code (like CUDA) by leveraging explicit knowledge of kernel implementations and hardware backends.

    Dynamic Graph Integration: Supports models with dynamic graph structures, allowing for flexible and efficient handling of graph-based neural networks (GNNs).

    Rule-Based Compilation: Uses an actor/DSL system to manage the code generation process, ensuring consistency and allowing for easy extension of new components, kernels, and optimization strategies.

    Separation of Concerns: Clearly separates the Model Architecture (.net files), Domain Knowledge (.unit files), and Code Generation Logic (.act files).

🚀 Getting Started

To get the ml-model-codegen project up and running, follow the instructions below.

Prerequisites

    (Placeholder for dependencies like C++ compiler, CUDA Toolkit, etc.)

Installation

    Clone the repository:
    Bash

git clone https://github.com/yourusername/ml-model-codegen.git
cd ml-model-codegen

Build the code generation tool:
Bash

    # Insert build command here (e.g., make or cmake)

Basic Usage

To generate code for a model (e.g., moe.net):
Bash

./codegen_tool --model-def input/moe.net --domain-unit units/net5.unit --actor-script actors/moe2.act --output-dir generated/

This will produce optimized CUDA files (like moe2.out) ready for compilation and deployment.

📖 Documentation

Detailed documentation on the framework's internal workings, the DSL specification, and the structure of model definition files is available separately:

    Core Framework Workings: Deep dive into the actor system, rule matching, and the code generation pipeline.

    Model Definition Reference: Full guide to defining models, blocks, parameters, and execution schedules using the .net file format.

    Domain Knowledge & Kernels: Explaining the components, kernels, and dynamic rules defined in the .unit files.

📄 License

This project is licensed under the MIT License - see the LICENSE.md file for details.
