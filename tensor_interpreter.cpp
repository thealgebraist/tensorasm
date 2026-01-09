#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <variant>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <Eigen/Dense>

// --- AST ---
struct Node { virtual ~Node() = default; };
struct Expression : Node {};
struct Statement : Node {};

struct NumberExpr : Expression { float val; NumberExpr(float v) : val(v) {} };
struct IdentExpr : Expression { std::string name; IdentExpr(std::string n) : name(n) {} };
struct IndexExpr : Expression {
    std::string name; std::unique_ptr<Expression> index;
    IndexExpr(std::string n, std::unique_ptr<Expression> i) : name(n), index(std::move(i)) {}
};
struct BinaryExpr : Expression {
    std::unique_ptr<Expression> left, right;
    std::string op;
    BinaryExpr(std::unique_ptr<Expression> l, std::string o, std::unique_ptr<Expression> r)
        : left(std::move(l)), op(o), right(std::move(r)) {}
};

struct VarDecl : Statement { std::string type, name; VarDecl(std::string t, std::string n) : type(t), name(n) {} };
struct SyncStmt : Statement { std::string name; SyncStmt(std::string n) : name(n) {} };
struct Assignment : Statement { std::string name; std::unique_ptr<Expression> expr; Assignment(std::string n, std::unique_ptr<Expression> e) : name(n), expr(std::move(e)) {} };
struct Intrinsic : Statement { std::string op; std::vector<std::unique_ptr<Expression>> args; Intrinsic(std::string o, std::vector<std::unique_ptr<Expression>> a) : op(o), args(std::move(a)) {} };
struct BatchLoop : Statement {
    std::string var; std::unique_ptr<Expression> start, end, step; std::vector<std::unique_ptr<Statement>> body;
    BatchLoop(std::string v, std::unique_ptr<Expression> s, std::unique_ptr<Expression> e, std::unique_ptr<Expression> st, std::vector<std::unique_ptr<Statement>> b)
        : var(v), start(std::move(s)), end(std::move(e)), step(std::move(st)), body(std::move(b)) {}
};

struct ConstDecl : Node {
    std::string name; int value;
    ConstDecl(std::string n, int v) : name(n), value(v) {}
};
struct TypeDef : Node {
    std::string name, prec, space, layout; std::vector<int> shape;
    TypeDef(std::string n, std::string p, std::string s, std::string l, std::vector<int> sh)
        : name(n), prec(p), space(s), layout(l), shape(sh) {}
};
struct TargetDecl : Node { std::string name; std::map<std::string, std::variant<int, std::string>> props; TargetDecl(std::string n) : name(n) {} };
struct Kernel : Node {
    std::string name; std::vector<std::pair<std::string, std::string>> params; std::vector<std::unique_ptr<Statement>> body; Kernel(std::string n) : name(n) {}
};
struct Program : Node { 
    std::unique_ptr<TargetDecl> target; 
    std::vector<std::unique_ptr<ConstDecl>> consts;
    std::vector<std::unique_ptr<TypeDef>> types; 
    std::vector<std::unique_ptr<Kernel>> kernels; 
};

// --- Lexer & Parser ---
enum class TokenType { Target, Type, Kernel, Batch, Tensor, F32, F16, BF16, Int8, Int32, Global, L1, Shared, TileReg, MMUL, LOAD, STORE, REDUCE, MADD, SOFTMAX, LOOKUP, SYNC, EXP, SQRT, TRANSPOSE, ACT, Const, Ident, Number,
    Equal, Semicolon, LParen, RParen, LBrace, RBrace, LAngle, RAngle, Comma, DotDot, Step, LBracket, RBracket, Plus, Minus, Star, Slash, Eof };
struct Token { TokenType type; std::string val; int line; };

class Lexer {
    std::string src; size_t pos = 0; int line = 1;
public:
    Lexer(std::string s) : src(std::move(s)) {}
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
            if (curr().type == TokenType::LBracket) { match(TokenType::LBracket); auto idx = parse_expr(); match(TokenType::RBracket); return std::make_unique<IndexExpr>(name, std::move(idx)); }
            return std::make_unique<IdentExpr>(name);
        }
        if (curr().type == TokenType::LParen) { match(TokenType::LParen); auto e = parse_expr(); match(TokenType::RParen); return e; }
        throw std::runtime_error("Invalid expression");
    }
    std::unique_ptr<Expression> parse_mul() {
        auto left = parse_primary();
        while (curr().type == TokenType::Star) { match(TokenType::Star); auto right = parse_primary(); left = std::make_unique<BinaryExpr>(std::move(left), "*", std::move(right)); }
        return left;
    }
    std::unique_ptr<Expression> parse_expr() {
        auto left = parse_mul();
        while (curr().type == TokenType::Plus || curr().type == TokenType::Minus) { auto op = (match(curr().type).type == TokenType::Plus) ? "+" : "-"; auto right = parse_mul(); left = std::make_unique<BinaryExpr>(std::move(left), op, std::move(right)); }
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
            match(TokenType::SYNC); match(TokenType::LParen); auto name = match(TokenType::Ident).val; match(TokenType::RParen); match(TokenType::Semicolon); return std::make_unique<SyncStmt>(name);
        } else if (curr().type == TokenType::MMUL || curr().type == TokenType::LOAD || curr().type == TokenType::STORE || curr().type == TokenType::REDUCE ||
                   curr().type == TokenType::MADD || curr().type == TokenType::SOFTMAX || curr().type == TokenType::LOOKUP ||
                   curr().type == TokenType::EXP || curr().type == TokenType::SQRT || curr().type == TokenType::TRANSPOSE || curr().type == TokenType::ACT) {
            auto op = match(curr().type).val; match(TokenType::LParen);
            std::vector<std::unique_ptr<Expression>> args;
            if (curr().type != TokenType::RParen) { args.push_back(parse_expr()); while (curr().type == TokenType::Comma) { match(TokenType::Comma); args.push_back(parse_expr()); } }
            match(TokenType::RParen); match(TokenType::Semicolon);
            return std::make_unique<Intrinsic>(op, std::move(args));
        } else {
            auto first = match(TokenType::Ident).val;
            if (curr().type == TokenType::Ident) { auto second = match(TokenType::Ident).val; match(TokenType::Semicolon); return std::make_unique<VarDecl>(first, second); }
            match(TokenType::Equal); auto expr = parse_expr(); match(TokenType::Semicolon); return std::make_unique<Assignment>(first, std::move(expr));
        }
    }

public:
    Parser(std::vector<Token> t) : tokens(std::move(t)) {}
    std::unique_ptr<Program> parse() {
        auto prog = std::make_unique<Program>();
        while (curr().type != TokenType::Eof) {
            if (curr().type == TokenType::Target) {
                match(TokenType::Target); auto name = match(TokenType::Ident).val; match(TokenType::LBrace);
                auto t = std::make_unique<TargetDecl>(name);
                while (curr().type != TokenType::RBrace) {
                    auto prop = match(TokenType::Ident).val; match(TokenType::Equal);
                    if (curr().type == TokenType::Number) t->props[prop] = std::stoi(match(TokenType::Number).val);
                    else t->props[prop] = match(TokenType::Ident).val;
                    match(TokenType::Semicolon);
                }
                match(TokenType::RBrace); prog->target = std::move(t);
            } else if (curr().type == TokenType::Const) {
                match(TokenType::Const); auto name = match(TokenType::Ident).val; match(TokenType::Equal); int val = std::stoi(match(TokenType::Number).val); match(TokenType::Semicolon);
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
                match(TokenType::Kernel); auto name = match(TokenType::Ident).val; match(TokenType::LParen);
                auto k = std::make_unique<Kernel>(name);
                if (curr().type != TokenType::RParen) { do { if (curr().type == TokenType::Comma) match(TokenType::Comma); auto ptype = match(TokenType::Ident).val; auto pname = match(TokenType::Ident).val; k->params.push_back({ptype, pname}); } while (curr().type == TokenType::Comma); }
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
            if (it != var_map.end()) return it->second;
            return nullptr;
        } else if (auto b = dynamic_cast<BinaryExpr*>(e)) {
            analyze_expr(b->left.get());
            analyze_expr(b->right.get());
            return nullptr;
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
        } else if (is_matrix(a) && is_vector(b)) {
            if (a->shape[1] != b->shape[0]) throw std::runtime_error("MMUL shape mismatch: A.cols must match B.size");
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
                    // Relaxed check for indexing
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
            if (it == type_map.end()) throw std::runtime_error("Unknown type: " + v->type);
            if (var_map.count(v->name)) throw std::runtime_error("Variable redefined: " + v->name);
            var_map[v->name] = it->second;
        } else if (auto a = dynamic_cast<Assignment*>(s)) {
            if (var_map.find(a->name) == var_map.end()) throw std::runtime_error("Undefined variable: " + a->name);
            analyze_expr(a->expr.get());
        } else if (auto i = dynamic_cast<Intrinsic*>(s)) {
            for (auto& arg : i->args) analyze_expr(arg.get());
            if (i->op == "MMUL") check_mmul(i);
            else if (i->op == "MADD") check_madd(i);
            else if (i->op == "LOAD") check_load_store(i, true);
            else if (i->op == "STORE") check_load_store(i, false);
            else if (i->op == "SOFTMAX") check_softmax(i);
            else if (i->op == "LOOKUP") check_lookup(i);
            else if (i->op == "EXP" || i->op == "SQRT") {
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
            if (var_map.find(sy->name) == var_map.end()) throw std::runtime_error("Undefined sync identifier: " + sy->name);
        }
    }
};

// --- Interpreter ---
struct Value {
    std::variant<Eigen::MatrixXf, Eigen::VectorXf, float> data;
};

struct Runtime {
    std::map<std::string, TypeDef*> types;
    std::map<std::string, TypeDef*> var_types;
    std::map<std::string, Value> env;
    std::map<std::string, int> consts;

    bool is_matrix(const Value& v) const { return std::holds_alternative<Eigen::MatrixXf>(v.data); }
    bool is_vector(const Value& v) const { return std::holds_alternative<Eigen::VectorXf>(v.data); }

    float eval_scalar(Expression* e) {
        auto val = eval_expr(e);
        if (auto f = std::get_if<float>(&val)) return *f;
        if (auto vp = std::get_if<Value*>(&val)) {
            if (std::holds_alternative<float>((*vp)->data)) return std::get<float>((*vp)->data);
            if (std::holds_alternative<Eigen::VectorXf>((*vp)->data)) {
                auto& vec = std::get<Eigen::VectorXf>((*vp)->data);
                if (vec.size() == 1) return vec(0);
            }
            if (std::holds_alternative<Eigen::MatrixXf>((*vp)->data)) {
                auto& mat = std::get<Eigen::MatrixXf>((*vp)->data);
                if (mat.size() == 1) return mat.data()[0];
            }
        }
        throw std::runtime_error("Expression is not a scalar");
    }

    Value& get_ref(const std::string& name) {
        auto it = env.find(name);
        if (it != env.end()) return it->second;
        auto cit = consts.find(name);
        if (cit != consts.end()) {
            env[name] = Value{float(cit->second)};
            var_types[name] = nullptr;
            return env[name];
        }
        throw std::runtime_error("Unknown variable: " + name);
    }

    std::variant<float, Value*> eval_expr(Expression* e) {
        if (auto n = dynamic_cast<NumberExpr*>(e)) return float(n->val);
        if (auto id = dynamic_cast<IdentExpr*>(e)) return &get_ref(id->name);
        if (auto idx = dynamic_cast<IndexExpr*>(e)) {
            return &get_ref(idx->name);
        }
        if (auto b = dynamic_cast<BinaryExpr*>(e)) {
            auto lv = eval_expr(b->left.get());
            auto rv = eval_expr(b->right.get());
            if (std::holds_alternative<float>(lv) && std::holds_alternative<float>(rv)) {
                float l = std::get<float>(lv), r = std::get<float>(rv);
                if (b->op == "+") return l + r;
                if (b->op == "-") return l - r;
                if (b->op == "*") return l * r;
            }
            Value* res = new Value();
            auto apply_op = [&](const auto& l, const auto& r) {
                auto to_array = [](const auto& x) {
                    if constexpr (std::is_scalar_v<std::decay_t<decltype(x)>>) return x;
                    else return x.array();
                };
                auto la = to_array(l);
                auto ra = to_array(r);
                if (b->op == "+") return (la + ra).matrix().eval();
                if (b->op == "-") return (la - ra).matrix().eval();
                if (b->op == "*") return (la * ra).matrix().eval();
                throw std::runtime_error("Unknown op");
            };
            auto get_data = [](std::variant<float, Value*> v) -> std::variant<float, Eigen::MatrixXf*, Eigen::VectorXf*> {
                if (auto f = std::get_if<float>(&v)) return *f;
                auto vp = std::get<Value*>(v);
                if (std::holds_alternative<Eigen::MatrixXf>(vp->data)) return &std::get<Eigen::MatrixXf>(vp->data);
                if (std::holds_alternative<Eigen::VectorXf>(vp->data)) return &std::get<Eigen::VectorXf>(vp->data);
                return std::get<float>(vp->data);
            };
            auto ld = get_data(lv), rd = get_data(rv);
            if (std::holds_alternative<Eigen::MatrixXf*>(ld) && std::holds_alternative<float>(rd)) res->data = apply_op(*std::get<Eigen::MatrixXf*>(ld), std::get<float>(rd));
            else if (std::holds_alternative<float>(ld) && std::holds_alternative<Eigen::MatrixXf*>(rd)) res->data = apply_op(std::get<float>(ld), *std::get<Eigen::MatrixXf*>(rd));
            else if (std::holds_alternative<Eigen::MatrixXf*>(ld) && std::holds_alternative<Eigen::MatrixXf*>(rd)) res->data = apply_op(*std::get<Eigen::MatrixXf*>(ld), *std::get<Eigen::MatrixXf*>(rd));
            else if (std::holds_alternative<Eigen::VectorXf*>(ld) && std::holds_alternative<float>(rd)) res->data = apply_op(*std::get<Eigen::VectorXf*>(ld), std::get<float>(rd));
            else if (std::holds_alternative<float>(ld) && std::holds_alternative<Eigen::VectorXf*>(rd)) res->data = apply_op(std::get<float>(ld), *std::get<Eigen::VectorXf*>(rd));
            else if (std::holds_alternative<Eigen::VectorXf*>(ld) && std::holds_alternative<Eigen::VectorXf*>(rd)) res->data = apply_op(*std::get<Eigen::VectorXf*>(ld), *std::get<Eigen::VectorXf*>(rd));
            return res;
        }
        throw std::runtime_error("Bad expression");
    }

    TypeDef* type_for_expr(Expression* e) const {
        if (auto id = dynamic_cast<IdentExpr*>(e)) { auto it = var_types.find(id->name); if (it != var_types.end()) return it->second; }
        else if (auto idx = dynamic_cast<IndexExpr*>(e)) { auto it = var_types.find(idx->name); if (it != var_types.end()) return it->second; }
        return nullptr;
    }

    void warn_tile(const std::string& op, Expression* e) const {
        if (auto td = type_for_expr(e)) if (td && td->space != "TileReg") std::cerr << "Warning: " << op << " operand is in " << td->space << ", expected TileReg" << std::endl;
    }

    void make_value(const TypeDef* td, const std::string& name, bool random=false, bool zero=false) {
        Value v;
        if (td->shape.size() == 2) { Eigen::MatrixXf m(td->shape[0], td->shape[1]); if (random) m.setConstant(0.5f); else if (zero) m.setZero(); v.data = std::move(m); }
        else { Eigen::VectorXf vec(td->shape[0]); if (random) vec.setConstant(0.5f); else if (zero) vec.setZero(); v.data = std::move(vec); }
        env[name] = std::move(v); var_types[name] = const_cast<TypeDef*>(td);
    }

    float* get_ptr(Expression* e) {
        if (auto id = dynamic_cast<IdentExpr*>(e)) {
            auto& v = get_ref(id->name);
            if (is_matrix(v)) return std::get<Eigen::MatrixXf>(v.data).data();
            if (is_vector(v)) return std::get<Eigen::VectorXf>(v.data).data();
        } else if (auto idx = dynamic_cast<IndexExpr*>(e)) {
            auto& v = get_ref(idx->name);
            int i = (int)eval_scalar(idx->index.get());
            auto it = var_types.find(idx->name);
            TypeDef* td = (it != var_types.end()) ? it->second : nullptr;
            if (td && td->shape.size() == 2) {
                if (is_matrix(v)) return std::get<Eigen::MatrixXf>(v.data).data() + i * td->shape[1];
            } else {
                if (is_matrix(v)) return std::get<Eigen::MatrixXf>(v.data).data() + i;
                if (is_vector(v)) return std::get<Eigen::VectorXf>(v.data).data() + i;
            }
        }
        return nullptr;
    }

    void exec_intrinsic(Intrinsic* in) {
        if (in->op == "LOAD" || in->op == "STORE") {
            bool is_load = (in->op == "LOAD");
            Expression* dst_expr = in->args[0].get(); Expression* src_expr = in->args[1].get();
            float* src_ptr = get_ptr(src_expr); float* dst_ptr = get_ptr(dst_expr);
            if (is_load) {
                Value& d_v = get_ref(dynamic_cast<IdentExpr*>(dst_expr) ? dynamic_cast<IdentExpr*>(dst_expr)->name : dynamic_cast<IndexExpr*>(dst_expr)->name);
                if (src_ptr) {
                    if (is_matrix(d_v)) { auto& m = std::get<Eigen::MatrixXf>(d_v.data); m = Eigen::Map<Eigen::MatrixXf>(src_ptr, m.rows(), m.cols()); }
                    else if (is_vector(d_v)) { auto& v = std::get<Eigen::VectorXf>(d_v.data); v = Eigen::Map<Eigen::VectorXf>(src_ptr, v.size()); }
                } else {
                    float sval = eval_scalar(src_expr);
                    if (is_matrix(d_v)) std::get<Eigen::MatrixXf>(d_v.data).setConstant(sval);
                    else if (is_vector(d_v)) std::get<Eigen::VectorXf>(d_v.data).setConstant(sval);
                }
            } else { // STORE
                auto sv = eval_expr(src_expr);
                if (dst_ptr) {
                    if (auto s_v_ptr = std::get_if<Value*>(&sv)) {
                        Value* s_v = *s_v_ptr;
                        if (is_matrix(*s_v)) { auto& m = std::get<Eigen::MatrixXf>(s_v->data); Eigen::Map<Eigen::MatrixXf>(dst_ptr, m.rows(), m.cols()) = m; }
                        else if (is_vector(*s_v)) { auto& v = std::get<Eigen::VectorXf>(s_v->data); Eigen::Map<Eigen::VectorXf>(dst_ptr, v.size()) = v; }
                    } else if (auto sval = std::get_if<float>(&sv)) {
                        // How many elements to store? We need the type of the destination.
                        // But STORE(addr, scalar) is tricky. 
                        // If it's a direct IDENT or INDEX, we can get the type.
                        if (auto id = dynamic_cast<IdentExpr*>(dst_expr)) {
                            auto& d_v = get_ref(id->name);
                            if (is_matrix(d_v)) std::get<Eigen::MatrixXf>(d_v.data).setConstant(*sval);
                            else if (is_vector(d_v)) std::get<Eigen::VectorXf>(d_v.data).setConstant(*sval);
                        } else if (auto idx = dynamic_cast<IndexExpr*>(dst_expr)) {
                            *dst_ptr = *sval;
                        }
                    }
                }
            }
        } else if (in->op == "MMUL") {
            warn_tile("MMUL", in->args[0].get()); warn_tile("MMUL", in->args[1].get()); warn_tile("MMUL", in->args[2].get());
            auto acc = std::get<Value*>(eval_expr(in->args[0].get())), a = std::get<Value*>(eval_expr(in->args[1].get())), b = std::get<Value*>(eval_expr(in->args[2].get()));
            if (is_matrix(*a) && is_matrix(*b) && is_matrix(*acc)) std::get<Eigen::MatrixXf>(acc->data).noalias() += std::get<Eigen::MatrixXf>(a->data) * std::get<Eigen::MatrixXf>(b->data);
            else if (is_matrix(*a) && is_vector(*b) && is_vector(*acc)) std::get<Eigen::VectorXf>(acc->data).noalias() += std::get<Eigen::MatrixXf>(a->data) * std::get<Eigen::VectorXf>(b->data);
        } else if (in->op == "MADD") {
            auto acc = std::get<Value*>(eval_expr(in->args[0].get())), a = std::get<Value*>(eval_expr(in->args[1].get())), b = std::get<Value*>(eval_expr(in->args[2].get())), c = std::get<Value*>(eval_expr(in->args[3].get()));
            if (is_matrix(*acc) && is_matrix(*a) && is_matrix(*b) && is_matrix(*c)) std::get<Eigen::MatrixXf>(acc->data).noalias() += std::get<Eigen::MatrixXf>(a->data) * std::get<Eigen::MatrixXf>(b->data) + std::get<Eigen::MatrixXf>(c->data);
            else if (is_vector(*acc)) {
                auto& accv = std::get<Eigen::VectorXf>(acc->data);
                if (is_matrix(*a) && is_vector(*b)) accv.noalias() += std::get<Eigen::MatrixXf>(a->data) * std::get<Eigen::VectorXf>(b->data);
                else if (is_vector(*a) && is_vector(*b)) accv.array() += std::get<Eigen::VectorXf>(a->data).array() * std::get<Eigen::VectorXf>(b->data).array();
                if (is_vector(*c)) accv += std::get<Eigen::VectorXf>(c->data);
                else if (std::holds_alternative<float>(c->data)) accv.array() += std::get<float>(c->data);
            }
        } else if (in->op == "LOOKUP") {
            std::cerr << "Debug: LOOKUP start" << std::endl;
            auto v0 = eval_expr(in->args[0].get());
            auto v1 = eval_expr(in->args[1].get());
            auto dst = std::get<Value*>(v0);
            auto table = std::get<Value*>(v1);
            std::cerr << "Debug: LOOKUP dst and table resolved" << std::endl;
            int idx = static_cast<int>(eval_scalar(in->args[2].get()));
            std::cerr << "Debug: LOOKUP idx=" << idx << std::endl;
            if (!std::holds_alternative<Eigen::MatrixXf>(table->data)) {
                std::cerr << "Error: LOOKUP table is not a matrix" << std::endl;
                throw std::runtime_error("LOOKUP table is not a matrix");
            }
            const auto& tbl = std::get<Eigen::MatrixXf>(table->data);
            if (is_vector(*dst)) std::get<Eigen::VectorXf>(dst->data) = tbl.row(idx).transpose();
            else if (is_matrix(*dst)) std::get<Eigen::MatrixXf>(dst->data).row(0) = tbl.row(idx);
            std::cerr << "Debug: LOOKUP end" << std::endl;
            else if (i->op == "REDUCE") {
                if (i->args.size() != 2) throw std::runtime_error("REDUCE requires 2 arguments (dst, src)");
                auto dst = std::get<Value*>(eval_expr(i->args[0].get()));
                auto src = std::get<Value*>(eval_expr(i->args[1].get()));
                float sum = 0.0f;
                if (is_matrix(*src)) sum = std::get<Eigen::MatrixXf>(src->data).sum();
                else if (is_vector(*src)) sum = std::get<Eigen::VectorXf>(src->data).sum();
                else sum = std::get<float>(src->data); // Should not happen with semantic checks
                
                if (is_matrix(*dst)) std::get<Eigen::MatrixXf>(dst->data).setConstant(sum);
                else if (is_vector(*dst)) std::get<Eigen::VectorXf>(dst->data).setConstant(sum);
                else dst->data = sum;
            } else if (i->op == "SOFTMAX") {
            auto v = std::get<Value*>(eval_expr(in->args[0].get()));
            if (is_matrix(*v)) std::get<Eigen::MatrixXf>(v->data) = std::get<Eigen::MatrixXf>(v->data).array().exp().matrix();
            else if (is_vector(*v)) std::get<Eigen::VectorXf>(v->data) = std::get<Eigen::VectorXf>(v->data).array().exp().matrix();
        } else if (in->op == "SQRT") {
            auto v = std::get<Value*>(eval_expr(in->args[0].get()));
            if (is_matrix(*v)) std::get<Eigen::MatrixXf>(v->data) = std::get<Eigen::MatrixXf>(v->data).array().sqrt().matrix();
            else if (is_vector(*v)) std::get<Eigen::VectorXf>(v->data) = std::get<Eigen::VectorXf>(v->data).array().sqrt().matrix();
        } else if (in->op == "TRANSPOSE") {
            auto dst = std::get<Value*>(eval_expr(in->args[0].get()));
            auto src = std::get<Value*>(eval_expr(in->args[1].get()));
            if (is_matrix(*src)) std::get<Eigen::MatrixXf>(dst->data) = std::get<Eigen::MatrixXf>(src->data).transpose();
        } else if (in->op == "ACT") {
            auto v = std::get<Value*>(eval_expr(in->args[0].get()));
            int type = static_cast<int>(eval_scalar(in->args[1].get()));
            auto apply = [&](auto& mat) {
                if (type == 1) { // GELU
                    mat = 0.5f * mat.array() * (1.0f + (0.7978845608f * (mat.array() + 0.044715f * mat.array().cube())).tanh()).matrix();
                } else if (type == 2) { // SILU
                    mat = mat.array() / (1.0f + (-mat.array()).exp()).matrix();
                }
            };
            if (is_matrix(*v)) apply(std::get<Eigen::MatrixXf>(v->data));
            else if (is_vector(*v)) apply(std::get<Eigen::VectorXf>(v->data));
        } else if (in->op == "SOFTMAX") {
            auto v = std::get<Value*>(eval_expr(in->args[0].get()));
            auto& vec = std::get<Eigen::VectorXf>(v->data); vec = vec.array().exp() / vec.array().exp().sum();
        } else if (in->op == "EXP") {
            auto v = std::get<Value*>(eval_expr(in->args[0].get()));
            if (is_matrix(*v)) std::get<Eigen::MatrixXf>(v->data) = std::get<Eigen::MatrixXf>(v->data).array().exp().matrix();
            else if (is_vector(*v)) std::get<Eigen::VectorXf>(v->data) = std::get<Eigen::VectorXf>(v->data).array().exp().matrix();
        } else if (in->op == "SQRT") {
            auto v = std::get<Value*>(eval_expr(in->args[0].get()));
            if (is_matrix(*v)) std::get<Eigen::MatrixXf>(v->data) = std::get<Eigen::MatrixXf>(v->data).array().sqrt().matrix();
            else if (is_vector(*v)) std::get<Eigen::VectorXf>(v->data) = std::get<Eigen::VectorXf>(v->data).array().sqrt().matrix();
        } else if (in->op == "TRANSPOSE") {
            auto dst = std::get<Value*>(eval_expr(in->args[0].get()));
            auto src = std::get<Value*>(eval_expr(in->args[1].get()));
            std::get<Eigen::MatrixXf>(dst->data) = std::get<Eigen::MatrixXf>(src->data).transpose().eval();
        } else if (in->op == "ACT") {
            auto v = std::get<Value*>(eval_expr(in->args[0].get()));
            int type = (int)eval_scalar(in->args[1].get());
            auto apply_act = [&](auto& m) {
                auto a = m.array();
                if (type == 1) { // GELU
                    m = (0.5f * a * (1.0f + (0.7978845608f * (a + 0.044715f * a.pow(3))).tanh())).matrix();
                } else if (type == 2) { // SILU
                    m = (a / (1.0f + (-a).exp())).matrix();
                }
            };
            if (is_matrix(*v)) apply_act(std::get<Eigen::MatrixXf>(v->data));
            else if (is_vector(*v)) apply_act(std::get<Eigen::VectorXf>(v->data));
        }
    }

    void exec_stmt(Statement* s) {
        if (auto v = dynamic_cast<VarDecl*>(s)) { auto it = types.find(v->type); make_value(it->second, v->name, false, true); }
        else if (auto i = dynamic_cast<Intrinsic*>(s)) exec_intrinsic(i);
        else if (auto b = dynamic_cast<BatchLoop*>(s)) {
            int start = (int)eval_scalar(b->start.get()), end = (int)eval_scalar(b->end.get()), step = (int)eval_scalar(b->step.get());
            for (int x = start; x < end; x += step) { env[b->var] = Value{float(x)}; var_types[b->var] = nullptr; for (auto& st : b->body) exec_stmt(st.get()); }
        } else if (auto a = dynamic_cast<Assignment*>(s)) {
            auto& ref = get_ref(a->name); auto val = eval_expr(a->expr.get());
            if (auto f = std::get_if<float>(&val)) {
                if (std::holds_alternative<Eigen::MatrixXf>(ref.data)) std::get<Eigen::MatrixXf>(ref.data).setConstant(*f);
                else if (std::holds_alternative<Eigen::VectorXf>(ref.data)) std::get<Eigen::VectorXf>(ref.data).setConstant(*f);
                else ref.data = *f;
            } else ref.data = std::get<Value*>(val)->data;
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::ifstream file(argv[1]); std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    try {
        Lexer l(src); Parser p(l.tokenize()); auto prog = p.parse(); SemanticAnalyzer sa(prog.get()); sa.analyze();
        Runtime rt;
        for (auto& c : prog->consts) rt.consts[c->name] = c->value;
        for (auto& t : prog->types) rt.types[t->name] = t.get();
        for (auto& k : prog->kernels) {
            for (auto& par : k->params) {
                auto td_it = rt.types.find(par.first);
                if (td_it == rt.types.end()) throw std::runtime_error("Unknown type: " + par.first);
                TypeDef* td = td_it->second;
                rt.make_value(td, par.second, false, true); // start with zero
                if (td->space == "Global") {
                    std::string path = "weights/" + par.second + ".bin";
                    std::ifstream f(path, std::ios::binary);
                    if (f) {
                        Value& v = rt.env[par.second];
                        if (rt.is_matrix(v)) {
                            auto& m = std::get<Eigen::MatrixXf>(v.data);
                            f.read(reinterpret_cast<char*>(m.data()), m.size() * sizeof(float));
                        } else if (rt.is_vector(v)) {
                            auto& vec = std::get<Eigen::VectorXf>(v.data);
                            f.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(float));
                        }
                    } else {
                        // fallback to constant 0.5f if file missing
                        Value& v = rt.env[par.second];
                        if (rt.is_matrix(v)) std::get<Eigen::MatrixXf>(v.data).setConstant(0.5f);
                        else if (rt.is_vector(v)) std::get<Eigen::VectorXf>(v.data).setConstant(0.5f);
                    }
                }
            }
            for (auto& s : k->body) rt.exec_stmt(s.get());
        }
        std::cout << "Interpretation successful." << std::endl;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; return 1; }
    return 0;
}