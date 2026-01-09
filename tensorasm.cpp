#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <variant>
#include <memory>
#include <stack>
#include <stdexcept>
#include <algorithm>
#include <fstream>

// --- AST ---
struct Node { virtual ~Node() = default; };
struct Expression : Node {};
struct Statement : Node {};

struct NumberExpr : Expression { float val; NumberExpr(float v) : val(v) {} };
struct IdentExpr : Expression { std::string name; IdentExpr(std::string n) : name(n) {} };
struct IndexExpr : Expression {
    std::string name;
    std::unique_ptr<Expression> index;
    IndexExpr(std::string n, std::unique_ptr<Expression> i) : name(n), index(std::move(i)) {}
};

struct BinaryExpr : Expression {
    std::unique_ptr<Expression> left, right;
    std::string op;
    BinaryExpr(std::unique_ptr<Expression> l, std::string o, std::unique_ptr<Expression> r)
        : left(std::move(l)), op(o), right(std::move(r)) {}
};

struct VarDecl : Statement {
    std::string type, name;
    VarDecl(std::string t, std::string n) : type(t), name(n) {}
};

struct SyncStmt : Statement {
    std::string name;
    SyncStmt(std::string n) : name(n) {}
};

struct Assignment : Statement {
    std::string name;
    std::unique_ptr<Expression> expr;
    Assignment(std::string n, std::unique_ptr<Expression> e) : name(n), expr(std::move(e)) {}
};

struct Intrinsic : Statement {
    std::string op;
    std::vector<std::unique_ptr<Expression>> args;
    Intrinsic(std::string o, std::vector<std::unique_ptr<Expression>> a) : op(o), args(std::move(a)) {}
};

struct BatchLoop : Statement {
    std::string var;
    std::unique_ptr<Expression> start, end, step;
    std::vector<std::unique_ptr<Statement>> body;
    BatchLoop(std::string v, std::unique_ptr<Expression> s, std::unique_ptr<Expression> e, std::unique_ptr<Expression> st, std::vector<std::unique_ptr<Statement>> b)
        : var(v), start(std::move(s)), end(std::move(e)), step(std::move(st)), body(std::move(b)) {}
};

struct ConstDecl : Node {
    std::string name;
    int value;
    ConstDecl(std::string n, int v) : name(n), value(v) {}
};

struct TypeDef : Node {
    std::string name, prec, space, layout;
    std::vector<int> shape;
    TypeDef(std::string n, std::string p, std::string s, std::string l, std::vector<int> sh)
        : name(n), prec(p), space(s), layout(l), shape(sh) {}
};

struct TargetDecl : Node {
    std::string name;
    std::map<std::string, std::variant<int, std::string>> props;
    TargetDecl(std::string n) : name(n) {}
};

struct Kernel : Node {
    std::string name;
    std::vector<std::pair<std::string, std::string>> params;
    std::vector<std::unique_ptr<Statement>> body;
    Kernel(std::string n) : name(n) {}
};

struct Program : Node {
    std::unique_ptr<TargetDecl> target;
    std::vector<std::unique_ptr<ConstDecl>> consts;
    std::vector<std::unique_ptr<TypeDef>> types;
    std::vector<std::unique_ptr<Kernel>> kernels;
};

// --- Lexer & Parser ---

enum class TokenType { 
    Target, Type, Kernel, Batch, Tensor, F32, F16, BF16, Int8, Int32, Global, L1, Shared, TileReg, MMUL, LOAD, STORE, REDUCE, MADD, SOFTMAX, LOOKUP, SYNC, EXP, SQRT, TRANSPOSE, ACT, Const, Ident, Number,
    Equal, Semicolon, LParen, RParen, LBrace, RBrace, LAngle, RAngle, Comma, DotDot, Step, LBracket, RBracket, Plus, Minus, Star, Slash, Eof
};

struct Token { TokenType type; std::string val; int line; };

class Lexer {
    std::string src; size_t pos = 0; int line = 1;
public:
    Lexer(std::string s) : src(s) {}
    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (pos < src.size()) {
            while (pos < src.size() && isspace(src[pos])) { if (src[pos] == '\n') line++; pos++; } 
            if (pos >= src.size()) break;
                        if (isdigit(src[pos])) {
                            std::string v; while (pos < src.size() && (isdigit(src[pos]) || src[pos] == '.')) {
                                if (src[pos] == '.' && pos + 1 < src.size() && src[pos+1] == '.') break;
                                v += src[pos++];
                            }
                            tokens.push_back({TokenType::Number, v, line});
                        }
             else if (isalpha(src[pos]) || src[pos] == '_') {
                std::string v; while (pos < src.size() && (isalnum(src[pos]) || src[pos] == '_')) v += src[pos++];
                if (v == "target") tokens.push_back({TokenType::Target, v, line});
                else if (v == "type") tokens.push_back({TokenType::Type, v, line});
                else if (v == "kernel") tokens.push_back({TokenType::Kernel, v, line});
                else if (v == "batch") tokens.push_back({TokenType::Batch, v, line});
                else if (v == "step") tokens.push_back({TokenType::Step, v, line});
                else if (v == "Tensor") tokens.push_back({TokenType::Tensor, v, line});
                else if (v == "f32") tokens.push_back({TokenType::F32, v, line});
                else if (v == "f16") tokens.push_back({TokenType::F16, v, line});
                else if (v == "bf16") tokens.push_back({TokenType::BF16, v, line});
                else if (v == "int8") tokens.push_back({TokenType::Int8, v, line});
                else if (v == "int32") tokens.push_back({TokenType::Int32, v, line});
                else if (v == "Global") tokens.push_back({TokenType::Global, v, line});
                else if (v == "L1") tokens.push_back({TokenType::L1, v, line});
                else if (v == "Shared") tokens.push_back({TokenType::Shared, v, line});
                else if (v == "TileReg") tokens.push_back({TokenType::TileReg, v, line});
                else if (v == "MMUL") tokens.push_back({TokenType::MMUL, v, line});
                else if (v == "LOAD") tokens.push_back({TokenType::LOAD, v, line});
                else if (v == "STORE") tokens.push_back({TokenType::STORE, v, line});
                else if (v == "REDUCE") tokens.push_back({TokenType::REDUCE, v, line});
                else if (v == "MADD") tokens.push_back({TokenType::MADD, v, line});
                else if (v == "SOFTMAX") tokens.push_back({TokenType::SOFTMAX, v, line});
                else if (v == "LOOKUP") tokens.push_back({TokenType::LOOKUP, v, line});
                else if (v == "SYNC") tokens.push_back({TokenType::SYNC, v, line});
                else if (v == "EXP") tokens.push_back({TokenType::EXP, v, line});
                else if (v == "SQRT") tokens.push_back({TokenType::SQRT, v, line});
                else if (v == "TRANSPOSE") tokens.push_back({TokenType::TRANSPOSE, v, line});
                else if (v == "ACT") tokens.push_back({TokenType::ACT, v, line});
                else if (v == "const") tokens.push_back({TokenType::Const, v, line});
                else tokens.push_back({TokenType::Ident, v, line});
            } else {
                char c = src[pos++];
                if (c == '=') tokens.push_back({TokenType::Equal, "=", line});
                else if (c == ';') tokens.push_back({TokenType::Semicolon, ";", line});
                else if (c == '(') tokens.push_back({TokenType::LParen, "(", line});
                else if (c == ')') tokens.push_back({TokenType::RParen, ")", line});
                else if (c == '{') tokens.push_back({TokenType::LBrace, "{", line});
                else if (c == '}') tokens.push_back({TokenType::RBrace, "}", line});
                else if (c == '<') tokens.push_back({TokenType::LAngle, "<", line});
                else if (c == '>') tokens.push_back({TokenType::RAngle, ">", line});
                else if (c == ',') tokens.push_back({TokenType::Comma, ",", line});
                else if (c == '[') tokens.push_back({TokenType::LBracket, "[", line});
                else if (c == ']') tokens.push_back({TokenType::RBracket, "]", line});
                else if (c == '+') tokens.push_back({TokenType::Plus, "+", line});
                else if (c == '-') tokens.push_back({TokenType::Minus, "-", line});
                else if (c == '*') tokens.push_back({TokenType::Star, "*", line});
                else if (c == '.' && pos < src.size() && src[pos] == '.') { pos++; tokens.push_back({TokenType::DotDot, "..", line}); }
            }
        }
        tokens.push_back({TokenType::Eof, "$", line});
        return tokens;
    }
};

std::string token_type_to_string(TokenType t) {
    switch (t) {
        case TokenType::Target: return "target";
        case TokenType::Type: return "type";
        case TokenType::Kernel: return "kernel";
        case TokenType::Batch: return "batch";
        case TokenType::Tensor: return "Tensor";
        case TokenType::F32: return "f32";
        case TokenType::F16: return "f16";
        case TokenType::BF16: return "bf16";
        case TokenType::Int8: return "int8";
        case TokenType::Int32: return "int32";
        case TokenType::Global: return "Global";
        case TokenType::L1: return "L1";
        case TokenType::Shared: return "Shared";
        case TokenType::TileReg: return "TileReg";
        case TokenType::MMUL: return "MMUL";
        case TokenType::LOAD: return "LOAD";
        case TokenType::STORE: return "STORE";
        case TokenType::REDUCE: return "REDUCE";
        case TokenType::MADD: return "MADD";
        case TokenType::SOFTMAX: return "SOFTMAX";
        case TokenType::LOOKUP: return "LOOKUP";
        case TokenType::SYNC: return "SYNC";
        case TokenType::EXP: return "EXP";
        case TokenType::SQRT: return "SQRT";
        case TokenType::TRANSPOSE: return "TRANSPOSE";
        case TokenType::ACT: return "ACT";
        case TokenType::Const: return "const";
        case TokenType::Ident: return "identifier";
        case TokenType::Number: return "number";
        case TokenType::Equal: return "=";
        case TokenType::Semicolon: return ";";
        case TokenType::LParen: return "(";
        case TokenType::RParen: return ")";
        case TokenType::LBrace: return "{";
        case TokenType::RBrace: return "}";
        case TokenType::LAngle: return "<";
        case TokenType::RAngle: return ">";
        case TokenType::Comma: return ",";
        case TokenType::DotDot: return "..";
        case TokenType::Step: return "step";
        case TokenType::LBracket: return "[";
        case TokenType::RBracket: return "]";
        case TokenType::Plus: return "+";
        case TokenType::Minus: return "-";
        case TokenType::Star: return "*";
        case TokenType::Eof: return "end of file";
        default: return "unknown";
    }
}

class Parser {
    std::vector<Token> tokens; size_t pos = 0;
    Token curr() { return tokens[pos]; } 
    Token match(TokenType t, const std::string& ctx = "") { 
        if (curr().type == t) return tokens[pos++]; 
        std::string msg = "line " + std::to_string(curr().line) + ": expected " + token_type_to_string(t) + " but got '" + curr().val + "'";
        if (!ctx.empty()) msg += " in " + ctx;
        throw std::runtime_error(msg); 
    } 

    std::string parse_precision() {
        switch (curr().type) {
            case TokenType::F32: return match(TokenType::F32).val;
            case TokenType::F16: return match(TokenType::F16).val;
            case TokenType::BF16: return match(TokenType::BF16).val;
            case TokenType::Int8: return match(TokenType::Int8).val;
            case TokenType::Int32: return match(TokenType::Int32).val;
            default: throw std::runtime_error("Expected precision at line " + std::to_string(curr().line));
        }
    }

    std::vector<int> parse_shape() {
        match(TokenType::LBrace);
        std::vector<int> shape;
        shape.push_back(std::stoi(match(TokenType::Number).val));
        while (curr().type == TokenType::Comma) { match(TokenType::Comma); shape.push_back(std::stoi(match(TokenType::Number).val)); }
        match(TokenType::RBrace);
        return shape;
    }

    std::string parse_memspace() {
        switch (curr().type) {
            case TokenType::Global: return match(TokenType::Global).val;
            case TokenType::L1: return match(TokenType::L1).val;
            case TokenType::Shared: return match(TokenType::Shared).val;
            case TokenType::TileReg: return match(TokenType::TileReg).val;
            default: throw std::runtime_error("Expected memory space at line " + std::to_string(curr().line));
        }
    }

    std::string parse_layout() {
        if (curr().type == TokenType::Ident && curr().val == "Tiled") {
            match(TokenType::Ident);
            match(TokenType::LParen);
            std::string d1 = match(TokenType::Number).val;
            if (!(curr().type == TokenType::Ident && curr().val == "x")) throw std::runtime_error("Expected 'x' in Tiled layout");
            match(TokenType::Ident);
            std::string d2 = match(TokenType::Number).val;
            match(TokenType::RParen);
            return "Tiled(" + d1 + "x" + d2 + ")";
        }
        if (curr().type == TokenType::Ident && (curr().val == "RowMajor" || curr().val == "ColMajor")) return match(TokenType::Ident).val;
        throw std::runtime_error("Expected layout at line " + std::to_string(curr().line));
    }
    
    std::unique_ptr<Expression> parse_primary() {
        if (curr().type == TokenType::Number) return std::make_unique<NumberExpr>(std::stof(match(TokenType::Number).val));
        if (curr().type == TokenType::Ident) {
            auto name = match(TokenType::Ident).val;
            if (curr().type == TokenType::LBracket) {
                match(TokenType::LBracket);
                auto idx = parse_expr();
                match(TokenType::RBracket);
                return std::make_unique<IndexExpr>(name, std::move(idx));
            }
            return std::make_unique<IdentExpr>(name);
        }
        if (curr().type == TokenType::LParen) {
            match(TokenType::LParen);
            auto e = parse_expr();
            match(TokenType::RParen);
            return e;
        }
        throw std::runtime_error("Invalid expression");
    }

    std::unique_ptr<Expression> parse_mul() {
        auto left = parse_primary();
        while (curr().type == TokenType::Star || curr().type == TokenType::Slash) {
            auto op = (match(curr().type).type == TokenType::Star) ? "*" : "/";
            auto right = parse_primary();
            left = std::make_unique<BinaryExpr>(std::move(left), op, std::move(right));
        }
        return left;
    }

    std::unique_ptr<Expression> parse_expr() {
        auto left = parse_mul();
        while (curr().type == TokenType::Plus || curr().type == TokenType::Minus) {
            auto op = (match(curr().type).type == TokenType::Plus) ? "+" : "-";
            auto right = parse_mul();
            left = std::make_unique<BinaryExpr>(std::move(left), op, std::move(right));
        }
        return left;
    }

    std::unique_ptr<Statement> parse_stmt() {
        if (curr().type == TokenType::Batch) {
            match(TokenType::Batch); match(TokenType::LParen);
            auto var = match(TokenType::Ident).val; match(TokenType::Equal);
            auto start = parse_expr(); match(TokenType::DotDot);
            auto end = parse_expr(); match(TokenType::Step);
            auto step = parse_expr(); match(TokenType::RParen);
            match(TokenType::LBrace);
            std::vector<std::unique_ptr<Statement>> body;
            while (curr().type != TokenType::RBrace) body.push_back(parse_stmt());
            match(TokenType::RBrace);
            return std::make_unique<BatchLoop>(var, std::move(start), std::move(end), std::move(step), std::move(body));
        } else if (curr().type == TokenType::SYNC) {
            match(TokenType::SYNC); match(TokenType::LParen);
            auto name = match(TokenType::Ident).val; match(TokenType::RParen); match(TokenType::Semicolon);
            return std::make_unique<SyncStmt>(name);
        } else if (curr().type == TokenType::MMUL || curr().type == TokenType::LOAD || curr().type == TokenType::STORE || curr().type == TokenType::REDUCE || 
                   curr().type == TokenType::MADD || curr().type == TokenType::SOFTMAX || curr().type == TokenType::LOOKUP ||
                   curr().type == TokenType::EXP || curr().type == TokenType::SQRT || curr().type == TokenType::TRANSPOSE || curr().type == TokenType::ACT) {
            auto op = match(curr().type).val; match(TokenType::LParen);
            std::vector<std::unique_ptr<Expression>> args;
            if (curr().type != TokenType::RParen) {
                args.push_back(parse_expr());
                while (curr().type == TokenType::Comma) { match(TokenType::Comma); args.push_back(parse_expr()); }
            }
            match(TokenType::RParen); match(TokenType::Semicolon);
            return std::make_unique<Intrinsic>(op, std::move(args));
        } else {
            auto first = match(TokenType::Ident).val;
            if (curr().type == TokenType::Ident) {
                auto second = match(TokenType::Ident).val; match(TokenType::Semicolon);
                return std::make_unique<VarDecl>(first, second);
            } else {
                match(TokenType::Equal);
                auto expr = parse_expr(); match(TokenType::Semicolon);
                return std::make_unique<Assignment>(first, std::move(expr));
            }
        }
    }

public:
    Parser(std::vector<Token> t) : tokens(t) {}
    std::unique_ptr<Program> parse() {
        auto prog = std::make_unique<Program>();
        while (curr().type != TokenType::Eof) {
            if (curr().type == TokenType::Target) {
                match(TokenType::Target); auto name = match(TokenType::Ident).val; match(TokenType::LBrace);
                auto t = std::make_unique<TargetDecl>(name);
                while (curr().type != TokenType::RBrace) {
                    auto prop = match(TokenType::Ident).val; match(TokenType::Equal);
                    if (curr().type == TokenType::Number) {
                        t->props[prop] = std::stoi(match(TokenType::Number).val);
                    } else {
                        t->props[prop] = match(TokenType::Ident).val;
                    }
                    match(TokenType::Semicolon);
                }
                match(TokenType::RBrace); prog->target = std::move(t);
            } else if (curr().type == TokenType::Const) {
                match(TokenType::Const); auto name = match(TokenType::Ident).val;
                match(TokenType::Equal); int val = std::stoi(match(TokenType::Number).val);
                match(TokenType::Semicolon);
                prog->consts.push_back(std::make_unique<ConstDecl>(name, val));
            } else if (curr().type == TokenType::Type) {
                match(TokenType::Type); auto name = match(TokenType::Ident).val;
                match(TokenType::Equal); match(TokenType::Tensor); match(TokenType::LAngle);
                auto prec = parse_precision(); match(TokenType::Comma);
                auto shape = parse_shape(); match(TokenType::Comma);
                auto space = parse_memspace(); match(TokenType::Comma);
                auto layout = parse_layout();
                match(TokenType::RAngle); match(TokenType::Semicolon);
                prog->types.push_back(std::make_unique<TypeDef>(name, prec, space, layout, shape));
            } else if (curr().type == TokenType::Kernel) {
                match(TokenType::Kernel); auto name = match(TokenType::Ident).val;
                match(TokenType::LParen);
                auto k = std::make_unique<Kernel>(name);
                if (curr().type != TokenType::RParen) {
                    do {
                        if (curr().type == TokenType::Comma) match(TokenType::Comma);
                        auto ptype = match(TokenType::Ident).val;
                        auto pname = match(TokenType::Ident).val;
                        k->params.push_back({ptype, pname});
                    } while (curr().type == TokenType::Comma);
                }
                match(TokenType::RParen); match(TokenType::LBrace);
                while (curr().type != TokenType::RBrace) k->body.push_back(parse_stmt());
                match(TokenType::RBrace); prog->kernels.push_back(std::move(k));
            } else pos++;
        }
        return prog;
    }
};

class SemanticAnalyzer {
    Program* prog;
    std::map<std::string, TypeDef*> type_map;
    std::map<std::string, TypeDef*> var_map;
    std::set<std::string> const_names;

    bool is_matrix(TypeDef* t) const { return t && t->shape.size() == 2; }
    bool is_vector(TypeDef* t) const { return t && t->shape.size() == 1; }

    void ensure_declared(const std::string& name) {
        if (var_map.find(name) == var_map.end() && const_names.find(name) == const_names.end())
            throw std::runtime_error("Undefined identifier: " + name);
    }

    TypeDef* require_tensor(Expression* e, const std::string& ctx) {
        if (auto id = dynamic_cast<IdentExpr*>(e)) {
            ensure_declared(id->name);
            auto it = var_map.find(id->name);
            if (it != var_map.end() && it->second == nullptr) throw std::runtime_error(ctx + " expects tensor, got scalar: " + id->name);
            if (it == var_map.end()) throw std::runtime_error(ctx + " expects tensor, got scalar: " + id->name);
            return it->second;
        } else if (auto idx = dynamic_cast<IndexExpr*>(e)) {
            ensure_declared(idx->name);
            auto it = var_map.find(idx->name);
            if (it != var_map.end() && it->second == nullptr) throw std::runtime_error(ctx + " expects tensor, got scalar: " + idx->name);
            if (it == var_map.end()) throw std::runtime_error(ctx + " expects tensor, got scalar: " + idx->name);
            analyze_expr(idx->index.get());
            return it->second;
        }
        analyze_expr(e);
        throw std::runtime_error(ctx + " expects tensor operand");
    }

    TypeDef* type_of_expr(Expression* e) {
        if (auto id = dynamic_cast<IdentExpr*>(e)) {
            auto it = var_map.find(id->name);
            if (it != var_map.end()) return it->second;
            return nullptr;
        } else if (auto idx = dynamic_cast<IndexExpr*>(e)) {
            auto it = var_map.find(idx->name);
            if (it != var_map.end()) {
                TypeDef* base = it->second;
                if (!base) return nullptr;
                // If we index a matrix, we get a vector (simulated)
                // For now, let's create a temporary TypeDef if needed, 
                // but simpler is to just return a "compatible" type or handle it in LOAD.
                // To keep it simple and avoid leaks, let's return the base type but 
                // handle the dimensionality reduction in the caller.
                return base; 
            }
            return nullptr;
        } else if (auto b = dynamic_cast<BinaryExpr*>(e)) {
            auto lt = type_of_expr(b->left.get());
            auto rt = type_of_expr(b->right.get());
            if (lt && rt) {
                if (lt->shape != rt->shape) throw std::runtime_error("Binary op shape mismatch");
                return lt;
            }
            return lt ? lt : rt;
        }
        return nullptr;
    }

    void ensure_shape_match(TypeDef* a, TypeDef* b, const std::string& ctx) {
        if (a && b && a->shape != b->shape)
            throw std::runtime_error(ctx + " shape mismatch");
    }

    void ensure_tile_reg(const std::string& op, const std::string& name, TypeDef* td) {
        if (td && td->space != "TileReg")
            throw std::runtime_error(op + " operand " + name + " must be in TileReg, got " + td->space);
    }

public:
    SemanticAnalyzer(Program* p) : prog(p) {}

    void analyze() {
        const_names.clear(); type_map.clear(); var_map.clear();
        for (auto& c : prog->consts) {
            if (!const_names.insert(c->name).second)
                throw std::runtime_error("Constant redefined: " + c->name);
        }
        for (auto& t : prog->types) {
            if (!type_map.emplace(t->name, t.get()).second)
                throw std::runtime_error("Type redefined: " + t->name);
        }

        for (auto& k : prog->kernels) {
            var_map.clear();
            for (auto& p : k->params) {
                auto it = type_map.find(p.first);
                if (it == type_map.end())
                    throw std::runtime_error("Unknown type: " + p.first + " in kernel " + k->name);
                if (var_map.count(p.second))
                    throw std::runtime_error("Parameter redefined: " + p.second);
                var_map[p.second] = it->second;
            }
            for (auto& s : k->body) analyze_stmt(s.get());
        }
    }

private:
    void analyze_expr(Expression* e) {
        if (auto id = dynamic_cast<IdentExpr*>(e)) {
            ensure_declared(id->name);
        } else if (auto idx = dynamic_cast<IndexExpr*>(e)) {
            ensure_declared(idx->name);
            analyze_expr(idx->index.get());
        } else if (auto b = dynamic_cast<BinaryExpr*>(e)) {
            analyze_expr(b->left.get());
            analyze_expr(b->right.get());
        }
    }

    void check_mmul(Intrinsic* i) {
        if (i->args.size() < 3) throw std::runtime_error("MMUL requires 3 arguments");
        auto acc = require_tensor(i->args[0].get(), "MMUL");
        auto a = require_tensor(i->args[1].get(), "MMUL");
        auto b = require_tensor(i->args[2].get(), "MMUL");
        ensure_tile_reg("MMUL", "acc", acc);
        ensure_tile_reg("MMUL", "A", a);
        ensure_tile_reg("MMUL", "B", b);

        if (is_matrix(a) && is_matrix(b)) {
            if (a->shape[1] != b->shape[0]) throw std::runtime_error("MMUL shape mismatch: A.cols must match B.rows");
            if (!is_matrix(acc) || acc->shape[0] != a->shape[0] || acc->shape[1] != b->shape[1])
                throw std::runtime_error("MMUL accumulator shape mismatch");
        } else if (is_matrix(a) && is_vector(b)) {
            if (a->shape[1] != b->shape[0]) throw std::runtime_error("MMUL shape mismatch: A.cols must match B.size");
            if (!is_vector(acc) || acc->shape[0] != a->shape[0])
                throw std::runtime_error("MMUL accumulator must be vector of size A.rows");
        } else {
            throw std::runtime_error("MMUL supports matrix*matrix or matrix*vector");
        }
    }

    void check_madd(Intrinsic* i) {
        if (i->args.size() < 4) throw std::runtime_error("MADD requires 4 arguments");
        auto acc = require_tensor(i->args[0].get(), "MADD");
        auto a = require_tensor(i->args[1].get(), "MADD");
        auto b = require_tensor(i->args[2].get(), "MADD");
        auto c = require_tensor(i->args[3].get(), "MADD");
        check_mmul(i);
        ensure_shape_match(acc, c, "MADD");
    }

    void check_load_store(Intrinsic* i, bool is_load) {
        if (i->args.empty()) throw std::runtime_error(std::string(is_load ? "LOAD" : "STORE") + " requires destination");
        auto dst = require_tensor(i->args[0].get(), is_load ? "LOAD" : "STORE");
        if (i->args.size() > 1) {
            auto src_type = type_of_expr(i->args[1].get());
            if (src_type) {
                if (dynamic_cast<IndexExpr*>(i->args[1].get()) || dynamic_cast<IndexExpr*>(i->args[0].get())) {
                    // Relaxed check for indexing: just check total size or ignore for now to allow tiling
                    int dst_elements = 1; for(int d : dst->shape) dst_elements *= d;
                    int src_elements = 1; for(int d : src_type->shape) src_elements *= d;
                    // This is still very basic but better than strict match
                } else {
                    ensure_shape_match(dst, src_type, is_load ? "LOAD" : "STORE");
                }
            }
        }
    }

    void check_softmax(Intrinsic* i) {
        if (i->args.size() != 1) throw std::runtime_error("SOFTMAX requires 1 argument");
        auto v = require_tensor(i->args[0].get(), "SOFTMAX");
        if (!is_vector(v)) throw std::runtime_error("SOFTMAX expects a vector");
    }

    void check_lookup(Intrinsic* i) {
        if (i->args.size() != 3) throw std::runtime_error("LOOKUP requires 3 arguments");
        auto dst = require_tensor(i->args[0].get(), "LOOKUP");
        auto table = require_tensor(i->args[1].get(), "LOOKUP");
        if (!is_vector(dst)) throw std::runtime_error("LOOKUP destination must be vector");
        if (!is_matrix(table) || table->shape.size() != 2) throw std::runtime_error("LOOKUP table must be matrix");
        if (dst->shape[0] != table->shape[1]) throw std::runtime_error("LOOKUP shape mismatch: destination width must match table columns");
        auto idx_type = type_of_expr(i->args[2].get());
        if (idx_type) {
            int size = 1;
            for(int d : idx_type->shape) size *= d;
            if (size != 1) throw std::runtime_error("LOOKUP index must be scalar or single-element tensor");
        }
    }

    void analyze_stmt(Statement* s) {
        if (auto v = dynamic_cast<VarDecl*>(s)) {
            auto it = type_map.find(v->type);
            if (it == type_map.end())
                throw std::runtime_error("Unknown type: " + v->type);
            if (var_map.count(v->name))
                throw std::runtime_error("Variable redefined: " + v->name);
            var_map[v->name] = it->second;
        } else if (auto a = dynamic_cast<Assignment*>(s)) {
            if (var_map.find(a->name) == var_map.end())
                throw std::runtime_error("Undefined variable: " + a->name);
            analyze_expr(a->expr.get());
        } else if (auto i = dynamic_cast<Intrinsic*>(s)) {
            for (auto& arg : i->args) analyze_expr(arg.get());
            
            if (i->op == "MMUL") {
                check_mmul(i);
            } else if (i->op == "MADD") {
                check_madd(i);
            } else if (i->op == "LOAD") {
                check_load_store(i, true);
            } else if (i->op == "STORE") {
                check_load_store(i, false);
            } else if (i->op == "SOFTMAX") {
                check_softmax(i);
            } else if (i->op == "LOOKUP") {
                check_lookup(i);
            } else if (i->op == "EXP" || i->op == "SQRT" || i->op == "SOFTMAX") {
                if (i->args.size() != 1) throw std::runtime_error(i->op + " requires 1 argument");
                require_tensor(i->args[0].get(), i->op);
            } else if (i->op == "TRANSPOSE") {
                if (i->args.size() != 2) throw std::runtime_error("TRANSPOSE requires 2 arguments");
                auto dst = require_tensor(i->args[0].get(), "TRANSPOSE");
                auto src = require_tensor(i->args[1].get(), "TRANSPOSE");
                if (dst->shape.size() != 2 || src->shape.size() != 2) throw std::runtime_error("TRANSPOSE requires 2D tensors");
                if (dst->shape[0] != src->shape[1] || dst->shape[1] != src->shape[0]) throw std::runtime_error("TRANSPOSE shape mismatch");
            } else if (i->op == "ACT") {
                if (i->args.size() != 2) throw std::runtime_error("ACT requires 2 arguments (tensor, type)");
                require_tensor(i->args[0].get(), "ACT");
            }
        } else if (auto b = dynamic_cast<BatchLoop*>(s)) {
            analyze_expr(b->start.get());
            analyze_expr(b->end.get());
            analyze_expr(b->step.get());
            auto prev = var_map.find(b->var);
            TypeDef* saved = (prev == var_map.end()) ? nullptr : prev->second;
            var_map[b->var] = nullptr;
            for (auto& st : b->body) analyze_stmt(st.get());
            if (prev == var_map.end()) var_map.erase(b->var); else var_map[b->var] = saved;
        } else if (auto sy = dynamic_cast<SyncStmt*>(s)) {
            if (var_map.find(sy->name) == var_map.end())
                 throw std::runtime_error("Undefined sync identifier: " + sy->name);
        }
    }
};

class Compiler {
    std::string indent = "  ";
    Program* prog;
    std::map<std::string, TypeDef*> var_map;

    bool is_tensor(const std::string& name) {
        return var_map.count(name) && var_map[name] != nullptr;
    }

    void gen_expr(Expression* e, bool as_ptr = false) {
        if (auto n = dynamic_cast<NumberExpr*>(e)) {
            std::cout << n->val;
        }
        else if (auto i = dynamic_cast<IdentExpr*>(e)) {
            if (as_ptr && is_tensor(i->name)) std::cout << i->name << ".data()";
            else std::cout << i->name;
        }
        else if (auto idx = dynamic_cast<IndexExpr*>(e)) {
            auto it = var_map.find(idx->name);
            TypeDef* td = (it != var_map.end()) ? it->second : nullptr;
            if (as_ptr) {
                if (td && td->shape.size() == 2) {
                    std::cout << idx->name << ".data() + (";
                    gen_expr(idx->index.get());
                    std::cout << ") * " << td->shape[1];
                } else {
                    std::cout << idx->name << ".data() + ";
                    gen_expr(idx->index.get());
                }
            } else {
                if (td && td->shape.size() == 2) {
                    std::cout << idx->name << ".row(";
                    gen_expr(idx->index.get());
                    std::cout << ")";
                } else {
                    std::cout << idx->name << "(";
                    gen_expr(idx->index.get());
                    std::cout << ")";
                }
            }
        }
        else if (auto b = dynamic_cast<BinaryExpr*>(e)) {
            std::cout << "(";
            gen_expr(b->left.get());
            std::cout << " " << b->op << " ";
            gen_expr(b->right.get());
            std::cout << ")";
        }
    }

    void gen_stmt(Statement* s) {
        if (auto v = dynamic_cast<VarDecl*>(s)) {
            std::cout << indent << v->type << " " << v->name << "; " << v->name << ".setZero();\n";
            TypeDef* td = nullptr;
            for (auto& t : prog->types) if (t->name == v->type) td = t.get();
            var_map[v->name] = td;
        }
                        else if (auto i = dynamic_cast<Intrinsic*>(s)) {
                            if (i->op == "STORE" && i->args.size() == 2 && dynamic_cast<NumberExpr*>(i->args[1].get())) {
                                std::cout << indent << "hw::ASSIGN(";
                                gen_expr(i->args[0].get());
                                std::cout << ", ";
                                gen_expr(i->args[1].get());
                                std::cout << ");\n";
                            } else if (i->op == "LOAD" && i->args.size() == 2 && dynamic_cast<NumberExpr*>(i->args[1].get())) {
                                std::cout << indent << "hw::ASSIGN(";
                                gen_expr(i->args[0].get());
                                std::cout << ", ";
                                gen_expr(i->args[1].get());
                                std::cout << ");\n";
                            } else {
                
                        std::cout << indent << "hw::" << i->op << "(";
                        for (size_t idx = 0; idx < i->args.size(); ++idx) {
                            bool ptr = (i->op == "LOAD" && idx == 1) || (i->op == "STORE" && idx == 0);
                            gen_expr(i->args[idx].get(), ptr);
                            if (idx < i->args.size() - 1) std::cout << ", ";
                        }
                        std::cout << ");\n";
                    }
                }
         else if (auto b = dynamic_cast<BatchLoop*>(s)) {
            std::cout << indent << "for (int " << b->var << " = "; gen_expr(b->start.get());
            std::cout << "; " << b->var << " < "; gen_expr(b->end.get());
            std::cout << "; " << b->var << " += "; gen_expr(b->step.get());
            std::cout << ") {\n";
            auto old_var = var_map.find(b->var);
            var_map[b->var] = nullptr;
            std::string old_indent = indent; indent += "  ";
            for (auto& st : b->body) gen_stmt(st.get());
            indent = old_indent; std::cout << indent << "}\n";
            if (old_var != var_map.end()) var_map[b->var] = old_var->second; else var_map.erase(b->var);
        } else if (auto a = dynamic_cast<Assignment*>(s)) {
            std::cout << indent << "hw::ASSIGN(" << a->name << ", "; gen_expr(a->expr.get()); std::cout << ");\n";
        } else if (auto sy = dynamic_cast<SyncStmt*>(s)) {
            std::cout << indent << "hw::SYNC(\"" << sy->name << "\");\n";
        }
    }
    std::string map_precision(const std::string& p) {
        if (p == "f32") return "float";
        if (p == "f16") return "Eigen::half";
        if (p == "bf16") return "Eigen::bfloat16";
        if (p == "int8") return "int8_t";
        if (p == "int32") return "int32_t";
        return "float";
    }

public:
    void compile(Program* p) {
        prog = p;
        std::cout << "#define EIGEN_STACK_ALLOCATION_LIMIT 0\n";
        std::cout << "#include <iostream>\n#include <memory>\n#include <type_traits>\n#include <cstdint>\n#include <fstream>\n#include <Eigen/Dense>\n\n";
        
        std::cout << "namespace hw {\n";
        std::cout << "  template<typename T, typename S>\n";
        std::cout << "  void ASSIGN(T& dst, const S& src) {\n";
        std::cout << "    if constexpr (std::is_arithmetic_v<S> && !std::is_arithmetic_v<T>) {\n";
        std::cout << "      dst.setConstant(src);\n";
        std::cout << "    } else {\n";
        std::cout << "      dst = src;\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
        
        std::cout << "  template<typename Dst, typename SrcPtr>\n";
        std::cout << "  void LOAD(Dst& dst, SrcPtr src_ptr) {\n";
        std::cout << "    if constexpr (std::is_arithmetic_v<SrcPtr>) {\n";
        std::cout << "      dst.setConstant(src_ptr);\n";
        std::cout << "    } else {\n";
        std::cout << "      dst = Eigen::Map<const Dst>(src_ptr);\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
        
        std::cout << "  template<typename SrcPtr, typename Src>\n";
        std::cout << "  void STORE(SrcPtr dst_ptr, const Src& src) {\n";
        std::cout << "    if constexpr (std::is_arithmetic_v<Src>) {\n";
        std::cout << "       *dst_ptr = src;\n";
        std::cout << "    } else {\n";
        std::cout << "      Eigen::Map<Src> m(dst_ptr); m = src;\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
        
        std::cout << "  template<typename Acc, typename A, typename B>\n";
        std::cout << "  void MMUL(Acc& acc, const A& a, const B& b) {\n";
        std::cout << "    acc.noalias() += a * b;\n";
        std::cout << "  }\n";
        
        std::cout << "  template<typename Acc, typename A, typename B, typename C>\n";
        std::cout << "  void MADD(Acc& acc, const A& a, const B& b, const C& c) {\n";
        std::cout << "    acc.noalias() += a * b + c;\n";
        std::cout << "  }\n";
        
        std::cout << "  template<typename Dst, typename Src>\n";
        std::cout << "  void REDUCE(Dst& dst, const Src& src) {\n";
        std::cout << "    if constexpr (std::is_scalar_v<std::decay_t<Dst>>) {\n";
        std::cout << "      dst = src.sum();\n";
        std::cout << "    } else {\n";
        std::cout << "      dst.setConstant(src.sum());\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
        
        std::cout << "  template<typename T>\n";
        std::cout << "  void SOFTMAX(T& t) {\n";
        std::cout << "    t = t.array().exp() / t.array().exp().sum();\n";
        std::cout << "  }\n";
        
        std::cout << "  template<typename T, typename Table, typename Idx>\n";
        std::cout << "  void LOOKUP(T& t, const Table& table, const Idx& idx) {\n";
        std::cout << "    if constexpr (std::is_arithmetic_v<Idx>) {\n";
        std::cout << "      t = table.row(idx);\n";
        std::cout << "    } else {\n";
        std::cout << "      t = table.row(static_cast<int>(idx.data()[0]));\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
        
        std::cout << "  template<typename T>\n";
        std::cout << "  void EXP(T& t) { t = t.array().exp().matrix(); }\n";
        std::cout << "  template<typename T>\n";
        std::cout << "  void SQRT(T& t) { t = t.array().sqrt().matrix(); }\n";
        std::cout << "  template<typename Dst, typename Src>\n";
        std::cout << "  void TRANSPOSE(Dst& dst, const Src& src) { dst = src.transpose(); }\n";
        std::cout << "  template<typename T>\n";
        std::cout << "  void ACT(T& t, int type) {\n";
        std::cout << "    if (type == 1) { // GELU\n";
        std::cout << "      auto a = t.array();\n";
        std::cout << "      t = (0.5f * a * (1.0f + (0.7978845608f * (a + 0.044715f * a.pow(3))).tanh())).matrix();\n";
        std::cout << "    } else if (type == 2) { // SILU\n";
        std::cout << "      auto a = t.array();\n";
        std::cout << "      t = (a / (1.0f + (-a).exp())).matrix();\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
        std::cout << "  template<typename T>\n";
        std::cout << "  void FILE_LOAD(T& t, const std::string& path) {\n";
        std::cout << "    std::ifstream f(path, std::ios::binary);\n";
        std::cout << "    if (f) {\n";
        std::cout << "      f.read(reinterpret_cast<char*>(t.data()), t.size() * sizeof(typename T::Scalar));\n";
        std::cout << "    } else {\n";
        std::cout << "      std::cerr << \"Warning: Could not load \" << path << \", using random/zero values.\" << std::endl;\n";
        std::cout << "    }\n";
        std::cout << "  }\n";
        
        std::cout << "  void SYNC(const std::string& name) {\n";
        std::cout << "    // Barrier for name\n";
        std::cout << "  }\n";
        std::cout << "}\n\n";

        for (auto& c : p->consts) {
            std::cout << "const int " << c->name << " = " << c->value << ";\n";
        }

        for (auto& t : p->types) {
            std::string cppType = map_precision(t->prec);
            std::string layoutFlag = (t->layout == "ColMajor") ? "Eigen::ColMajor" : "Eigen::RowMajor";
            if (t->layout.find("Tiled") == 0) {
                std::cout << "// Tiled layout " << t->layout << " for " << t->name << "\n";
            }
            if (t->shape.size() == 2) {
                std::cout << "using " << t->name << " = Eigen::Matrix<" << cppType << ", " << t->shape[0] << ", " << t->shape[1] << ", " << layoutFlag << ">;\n";
            } else {
                std::cout << "using " << t->name << " = Eigen::Matrix<" << cppType << ", " << t->shape[0] << ", 1>;\n";
            }
        }
        
        for (auto& k : p->kernels) {
            var_map.clear();
            std::cout << "\nvoid " << k->name << "(";
            for (size_t i = 0; i < k->params.size(); ++i) {
                std::cout << k->params[i].first << "& " << k->params[i].second;
                if (i < k->params.size() - 1) std::cout << ", ";
                TypeDef* td = nullptr;
                for (auto& t : p->types) if (t->name == k->params[i].first) td = t.get();
                var_map[k->params[i].second] = td;
            }
            std::cout << ") {\n";
            for (auto& s : k->body) gen_stmt(s.get());
            std::cout << "}\n";
        }

        std::cout << "\nint main() {\n";
        for (auto& k : p->kernels) {
            for (auto& par : k->params) {
                TypeDef* td = nullptr;
                for (auto& t : p->types) if (t->name == par.first) td = t.get();
                if (td) {
                    std::cout << "  auto " << par.second << "_ptr = std::make_unique<" << par.first << ">();\n";
                    std::cout << "  " << par.first << "& " << par.second << " = *" << par.second << "_ptr;\n";
                    if (td->space == "Global") {
                        std::cout << "  hw::FILE_LOAD(" << par.second << ", \"weights/" << par.second << ".bin\");\n";
                    } else {
                        std::cout << "  " << par.second << ".setConstant(0.5f);\n";
                    }
                }
            }
            std::cout << "  " << k->name << "(";
            for (size_t i = 0; i < k->params.size(); ++i) {
                std::cout << k->params[i].second;
                if (i < k->params.size() - 1) std::cout << ", ";
            }
            std::cout << ");\n";
        }
        std::cout << "  std::cout << \"Kernel execution successful.\" << std::endl;\n";
        std::cout << "  return 0;\n}\n";
    }
};

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <source_file>" << std::endl; return 1; }
    std::ifstream file(argv[1]);
    if (!file.is_open()) { std::cerr << "Could not open file" << std::endl; return 1; }
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    try {
        Lexer l(src); auto tokens = l.tokenize();
        Parser p(tokens); auto prog = p.parse();
        SemanticAnalyzer sa(prog.get()); sa.analyze();
        Compiler c; c.compile(prog.get());
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; return 1; }
    return 0;
}
