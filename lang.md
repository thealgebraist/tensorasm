This LR(1) grammar defines the TensorCore DSL by strictly separating the Configuration/Type Phase from the Execution/Kernel Phase.To maintain simplicity and cache coherence, the grammar prohibits dynamic memory allocation. All tensor shapes must be resolvable at the time of kernel dispatch, typically by injecting values from a Hugging Face config.json.1. Top-Level ProductionsThe program is a collection of hardware definitions, constants, and kernels.EBNF<Program>      ::= <DeclList>
<DeclList>     ::= <Decl> <DeclList> | ε
<Decl>         ::= <TargetDecl> | <ConstDecl> | <TypeDef> | <KernelDecl>

<TargetDecl>   ::= "target" IDENT "{" <TargetProps> "}"
<TargetProps>  ::= <Prop> <TargetProps> | ε
<Prop>         ::= IDENT "=" IDENT ";" | IDENT "=" NUMBER ";"
2. Type System (Physical Tensors)The type system is the "brain" of the DSL. It encodes the hardware target and memory layout into the type itself.EBNF<TypeDef>      ::= "type" IDENT "=" <TensorType> ";"
<TensorType>   ::= "Tensor" "<" <Precision> "," <Shape> "," <MemSpace> "," <Layout> ">"

<Precision>    ::= "f32" | "f16" | "bf16" | "int8" | "int32"
<Shape>        ::= "{" <DimList> "}"
<DimList>      ::= NUMBER | NUMBER "," <DimList>

<MemSpace>     ::= "Global" | "L1" | "Shared" | "TileReg"
<Layout>       ::= "RowMajor" | "ColMajor" | "Tiled" "(" NUMBER "x" NUMBER ")"
3. Kernel & Statement GrammarKernels use a functional style. Variables are used instead of registers, but their MemSpace attribute determines if they are stored in DRAM or an AMX/CUDA register tile.EBNF<KernelDecl>   ::= "kernel" IDENT "(" <ParamList> ")" "{" <StmtList> "}"
<ParamList>    ::= <Param> | <Param> "," <ParamList> | ε
<Param>        ::= IDENT IDENT  /* e.g., WeightMatrix Wq */

<StmtList>     ::= <Stmt> <StmtList> | ε
<Stmt>         ::= <VarDecl> | <Assignment> | <BatchLoop> | <Intrinsic> | <Sync>

<VarDecl>      ::= IDENT IDENT ";"  /* e.g., Tile t1; */
<Assignment>   ::= IDENT "=" <Expr> ";"

<BatchLoop>    ::= "batch" "(" IDENT "=" <Expr> ".." <Expr> "step" <Expr> ")" "{" <StmtList> "}"

<Sync>         ::= "SYNC" "(" IDENT ")" ";"
4. Hardware Intrinsics (Instruction Set)The "Instructions" are high-level matrix operations. The LR(1) parser treats these as special terminals that the backend lowers into specific ISA codes (Neon, AMX, PTX).EBNF<Intrinsic>    ::= <Mnemonic> "(" <ArgList> ")" ";"
<Mnemonic>     ::= "LOAD" | "STORE" | "MMUL" | "MADD" | "REDUCE" | "SOFTMAX" | "LOOKUP" | "EXP" | "SQRT" | "TRANSPOSE" | "ACT"

<ArgList>      ::= <Expr> | <Expr> "," <ArgList>
<Expr>         ::= IDENT | NUMBER | IDENT "[" <Expr> "]" | <BinaryOp>
<BinaryOp>     ::= <Expr> "+" <Expr> | <Expr> "-" <Expr> | <Expr> "*" <Expr> | <Expr> "/" <Expr>
5. Tokenizer & Stable Diffusion ExtensionTo support tokenization and complex pipelines, the grammar includes a LOOKUP terminal for BPE tables and string handling.Example Parse Tree: MMUL(acc, w, x);Terminal: MMUL matches <Mnemonic>.ArgList: acc (IDENT), w (IDENT), x (IDENT) reduces to <ArgList>.Intrinsic: Combined into <Intrinsic>, then <Stmt>.Semantic Check: The compiler verifies that acc, w, and x all have the TileReg MemSpace in their type definitions. If w was Global, the parse would technically succeed, but the Semantic Analyzer would reject it for violating hardware affinity.Comparison with Traditional AssemblyUnlike standard x86 or ARM assembly:Variable Scope: Variables have lexical scope, but are statically mapped to physical hardware locations (e.g., Apple AMX $X, Y, Z$ tiles).Zero Ambiguity: Because every Tensor has a hardcoded shape, the parser can determine the exact unrolling factor for the batch loops at compile time.Would you like me to generate the LR(1) Action/Goto table for a subset of these productions to show how it handles the batch loop nesting?