from flask import Flask, request, jsonify, render_template
import ast
import re
import subprocess
import tempfile
import os
import sys
from typing import Dict, List, Any, Optional
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class PythonOptimizer:
    """Python code optimizer using AST transformations"""
    
    def __init__(self):
        self.optimizations_applied = []
    
    def optimize(self, code: str) -> Dict[str, Any]:
        """Main optimization function"""
        try:
            tree = ast.parse(code)
            original_tree = ast.dump(tree)
            
            # Apply optimizations
            tree = self._remove_dead_code(tree)
            tree = self._constant_folding(tree)
            tree = self._strength_reduction(tree)
            tree = self._loop_optimization(tree)
            tree = self._function_inlining(tree)
            
            # Convert back to code
            optimized_code = ast.unparse(tree)
            
            # Calculate metrics
            original_lines = len(code.strip().split('\n'))
            optimized_lines = len(optimized_code.strip().split('\n'))
            
            return {
                'success': True,
                'original_code': code,
                'optimized_code': optimized_code,
                'optimizations_applied': self.optimizations_applied,
                'metrics': {
                    'original_lines': original_lines,
                    'optimized_lines': optimized_lines,
                    'reduction_percentage': ((original_lines - optimized_lines) / original_lines * 100) if original_lines > 0 else 0
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_code': code
            }
    
    def _remove_dead_code(self, tree: ast.AST) -> ast.AST:
        """Remove unreachable code and unused variables"""
        class DeadCodeRemover(ast.NodeTransformer):
            def __init__(self):
                self.used_names = set()
                self.defined_names = set()
                self.optimizations = []
            
            def visit_FunctionDef(self, node):
                # Remove functions that are never called
                self.defined_names.add(node.name)
                return self.generic_visit(node)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    self.used_names.add(node.id)
                elif isinstance(node.ctx, ast.Store):
                    self.defined_names.add(node.id)
                return node
            
            def visit_If(self, node):
                # Remove if statements with constant False conditions
                if isinstance(node.test, ast.Constant) and not node.test.value:
                    self.optimizations.append("Removed dead if branch with constant False condition")
                    return None
                return self.generic_visit(node)
            
            def visit_While(self, node):
                # Remove while loops with constant False conditions
                if isinstance(node.test, ast.Constant) and not node.test.value:
                    self.optimizations.append("Removed dead while loop with constant False condition")
                    return None
                return self.generic_visit(node)
        
        transformer = DeadCodeRemover()
        new_tree = transformer.visit(tree)
        self.optimizations_applied.extend(transformer.optimizations)
        return new_tree
    
    def _constant_folding(self, tree: ast.AST) -> ast.AST:
        """Fold constant expressions"""
        class ConstantFolder(ast.NodeTransformer):
            def __init__(self):
                self.optimizations = []
            
            def visit_BinOp(self, node):
                node = self.generic_visit(node)
                
                if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                    try:
                        if isinstance(node.op, ast.Add):
                            result = node.left.value + node.right.value
                        elif isinstance(node.op, ast.Sub):
                            result = node.left.value - node.right.value
                        elif isinstance(node.op, ast.Mult):
                            result = node.left.value * node.right.value
                        elif isinstance(node.op, ast.Div):
                            if node.right.value != 0:
                                result = node.left.value / node.right.value
                            else:
                                return node
                        elif isinstance(node.op, ast.Mod):
                            if node.right.value != 0:
                                result = node.left.value % node.right.value
                            else:
                                return node
                        elif isinstance(node.op, ast.Pow):
                            result = node.left.value ** node.right.value
                        else:
                            return node
                        
                        self.optimizations.append(f"Folded constant expression: {node.left.value} {type(node.op).__name__} {node.right.value} = {result}")
                        return ast.Constant(value=result)
                    except:
                        return node
                
                return node
            
            def visit_UnaryOp(self, node):
                node = self.generic_visit(node)
                
                if isinstance(node.operand, ast.Constant):
                    try:
                        if isinstance(node.op, ast.UAdd):
                            result = +node.operand.value
                        elif isinstance(node.op, ast.USub):
                            result = -node.operand.value
                        elif isinstance(node.op, ast.Not):
                            result = not node.operand.value
                        else:
                            return node
                        
                        self.optimizations.append(f"Folded unary expression: {type(node.op).__name__} {node.operand.value} = {result}")
                        return ast.Constant(value=result)
                    except:
                        return node
                
                return node
        
        transformer = ConstantFolder()
        new_tree = transformer.visit(tree)
        self.optimizations_applied.extend(transformer.optimizations)
        return new_tree
    
    def _strength_reduction(self, tree: ast.AST) -> ast.AST:
        """Replace expensive operations with cheaper ones"""
        class StrengthReducer(ast.NodeTransformer):
            def __init__(self):
                self.optimizations = []
            
            def visit_BinOp(self, node):
                node = self.generic_visit(node)
                
                # Replace multiplication by 2 with addition
                if isinstance(node.op, ast.Mult):
                    if isinstance(node.right, ast.Constant) and node.right.value == 2:
                        self.optimizations.append("Replaced multiplication by 2 with addition")
                        return ast.BinOp(left=node.left, op=ast.Add(), right=node.left)
                    elif isinstance(node.left, ast.Constant) and node.left.value == 2:
                        self.optimizations.append("Replaced multiplication by 2 with addition")
                        return ast.BinOp(left=node.right, op=ast.Add(), right=node.right)
                
                # Replace division by power of 2 with right shift (conceptually)
                if isinstance(node.op, ast.Div):
                    if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                        if node.right.value > 0 and (node.right.value & (node.right.value - 1)) == 0:
                            # It's a power of 2, but we'll keep as division in Python
                            pass
                
                return node
        
        transformer = StrengthReducer()
        new_tree = transformer.visit(tree)
        self.optimizations_applied.extend(transformer.optimizations)
        return new_tree
    
    def _loop_optimization(self, tree: ast.AST) -> ast.AST:
        """Optimize loops"""
        class LoopOptimizer(ast.NodeTransformer):
            def __init__(self):
                self.optimizations = []
            
            def visit_For(self, node):
                node = self.generic_visit(node)
                
                # Loop unrolling for small constant ranges
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == 'range' and len(node.iter.args) == 1:
                        if isinstance(node.iter.args[0], ast.Constant) and isinstance(node.iter.args[0].value, int):
                            if node.iter.args[0].value <= 3:  # Small range
                                # Unroll the loop
                                unrolled_body = []
                                for i in range(node.iter.args[0].value):
                                    for stmt in node.body:
                                        # Replace loop variable with constant
                                        new_stmt = self._replace_var(stmt, node.target.id, i)
                                        unrolled_body.append(new_stmt)
                                
                                self.optimizations.append(f"Unrolled loop with {node.iter.args[0].value} iterations")
                                return unrolled_body
                
                return node
            
            def _replace_var(self, node, var_name, value):
                """Replace variable with constant value"""
                class VarReplacer(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id == var_name and isinstance(node.ctx, ast.Load):
                            return ast.Constant(value=value)
                        return node
                
                return VarReplacer().visit(node)
        
        transformer = LoopOptimizer()
        new_tree = transformer.visit(tree)
        
        # Flatten the tree if loop unrolling created nested lists
        class TreeFlattener(ast.NodeTransformer):
            def visit_Module(self, node):
                new_body = []
                for item in node.body:
                    if isinstance(item, list):
                        new_body.extend(item)
                    else:
                        new_body.append(item)
                node.body = new_body
                return node
        
        new_tree = TreeFlattener().visit(new_tree)
        self.optimizations_applied.extend(transformer.optimizations)
        return new_tree
    
    def _function_inlining(self, tree: ast.AST) -> ast.AST:
        """Inline simple functions"""
        class FunctionInliner(ast.NodeTransformer):
            def __init__(self):
                self.functions = {}
                self.optimizations = []
            
            def visit_Module(self, node):
                # First pass: collect simple functions
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if self._is_simple_function(item):
                            self.functions[item.name] = item
                
                return self.generic_visit(node)
            
            def visit_Call(self, node):
                node = self.generic_visit(node)
                
                if isinstance(node.func, ast.Name) and node.func.id in self.functions:
                    func_def = self.functions[node.func.id]
                    
                    # Simple inlining for functions with single return statement
                    if len(func_def.body) == 1 and isinstance(func_def.body[0], ast.Return):
                        if len(func_def.args.args) == len(node.args):
                            # Replace parameters with arguments
                            inlined_expr = func_def.body[0].value
                            for param, arg in zip(func_def.args.args, node.args):
                                inlined_expr = self._replace_param(inlined_expr, param.arg, arg)
                            
                            self.optimizations.append(f"Inlined function call: {node.func.id}")
                            return inlined_expr
                
                return node
            
            def _is_simple_function(self, func_def):
                """Check if function is simple enough to inline"""
                return (len(func_def.body) == 1 and 
                        isinstance(func_def.body[0], ast.Return) and
                        len(func_def.args.args) <= 3)
            
            def _replace_param(self, node, param_name, arg_value):
                """Replace parameter with argument value"""
                class ParamReplacer(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id == param_name and isinstance(node.ctx, ast.Load):
                            return arg_value
                        return node
                
                return ParamReplacer().visit(node)
        
        transformer = FunctionInliner()
        new_tree = transformer.visit(tree)
        self.optimizations_applied.extend(transformer.optimizations)
        return new_tree


class COptimizer:
    """C code optimizer using LLVM"""
    
    def __init__(self):
        self.optimizations_applied = []
    
    def optimize(self, code: str) -> Dict[str, Any]:
        """Optimize C code using LLVM"""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as c_file:
                c_file.write(code)
                c_file_path = c_file.name
            
            ll_file_path = c_file_path.replace('.c', '.ll')
            opt_ll_file_path = c_file_path.replace('.c', '_opt.ll')
            opt_c_file_path = c_file_path.replace('.c', '_opt.c')
            
            try:
                # Compile C to LLVM IR
                subprocess.run(['clang', '-S', '-emit-llvm', c_file_path, '-o', ll_file_path], 
                             check=True, capture_output=True)
                
                # Apply LLVM optimizations
                opt_cmd = [
                    'opt',
                    '-passes=default<O2>',  # Standard O2 optimizations
                    ll_file_path,
                    '-o', opt_ll_file_path
                ]
                subprocess.run(opt_cmd, check=True, capture_output=True)
                
                # Read original and optimized IR
                with open(ll_file_path, 'r') as f:
                    original_ir = f.read()
                
                with open(opt_ll_file_path, 'r') as f:
                    optimized_ir = f.read()
                
                # Generate optimized C code (simplified)
                optimized_c_code = self._generate_optimized_c(code, original_ir, optimized_ir)
                
                # Calculate metrics
                original_lines = len(code.strip().split('\n'))
                optimized_lines = len(optimized_c_code.strip().split('\n'))
                
                # Analyze optimizations applied
                self._analyze_optimizations(original_ir, optimized_ir)
                
                return {
                    'success': True,
                    'original_code': code,
                    'optimized_code': optimized_c_code,
                    'original_ir': original_ir,
                    'optimized_ir': optimized_ir,
                    'optimizations_applied': self.optimizations_applied,
                    'metrics': {
                        'original_lines': original_lines,
                        'optimized_lines': optimized_lines,
                        'reduction_percentage': ((original_lines - optimized_lines) / original_lines * 100) if original_lines > 0 else 0
                    }
                }
                
            finally:
                # Clean up temporary files
                for file_path in [c_file_path, ll_file_path, opt_ll_file_path, opt_c_file_path]:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                        
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': f"LLVM compilation error: {e.stderr.decode() if e.stderr else str(e)}",
                'original_code': code
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_code': code
            }
    
    def _generate_optimized_c(self, original_code: str, original_ir: str, optimized_ir: str) -> str:
        """Generate optimized C code based on IR analysis"""
        # This is a simplified approach - in practice, you'd need more sophisticated IR-to-C translation
        optimized_code = original_code
        
        # Apply some basic optimizations that we can detect
        optimized_code = self._apply_constant_folding(optimized_code)
        optimized_code = self._apply_strength_reduction(optimized_code)
        optimized_code = self._remove_dead_code(optimized_code)
        
        return optimized_code
    
    def _apply_constant_folding(self, code: str) -> str:
        """Apply constant folding optimizations"""
        # Simple constant folding patterns
        patterns = [
            (r'(\d+)\s*\+\s*(\d+)', lambda m: str(int(m.group(1)) + int(m.group(2)))),
            (r'(\d+)\s*\-\s*(\d+)', lambda m: str(int(m.group(1)) - int(m.group(2)))),
            (r'(\d+)\s*\*\s*(\d+)', lambda m: str(int(m.group(1)) * int(m.group(2)))),
        ]
        
        for pattern, replacement in patterns:
            old_code = code
            code = re.sub(pattern, replacement, code)
            if code != old_code:
                self.optimizations_applied.append("Applied constant folding")
        
        return code
    
    def _apply_strength_reduction(self, code: str) -> str:
        """Apply strength reduction optimizations"""
        # Replace multiplication by 2 with left shift
        old_code = code
        code = re.sub(r'(\w+)\s*\*\s*2\b', r'\1 << 1', code)
        code = re.sub(r'2\s*\*\s*(\w+)', r'\1 << 1', code)
        
        # Replace division by power of 2 with right shift
        code = re.sub(r'(\w+)\s*/\s*2\b', r'\1 >> 1', code)
        code = re.sub(r'(\w+)\s*/\s*4\b', r'\1 >> 2', code)
        code = re.sub(r'(\w+)\s*/\s*8\b', r'\1 >> 3', code)
        
        if code != old_code:
            self.optimizations_applied.append("Applied strength reduction")
        
        return code
    
    def _remove_dead_code(self, code: str) -> str:
        """Remove obvious dead code"""
        lines = code.split('\n')
        new_lines = []
        
        for line in lines:
            # Remove lines that are obviously dead (like if(0))
            if 'if(0)' not in line and 'if (0)' not in line:
                new_lines.append(line)
            else:
                self.optimizations_applied.append("Removed dead code")
        
        return '\n'.join(new_lines)
    
    def _analyze_optimizations(self, original_ir: str, optimized_ir: str):
        """Analyze what optimizations were applied by comparing IR"""
        original_lines = original_ir.count('\n')
        optimized_lines = optimized_ir.count('\n')
        
        if optimized_lines < original_lines:
            self.optimizations_applied.append(f"LLVM optimizations reduced IR from {original_lines} to {optimized_lines} lines")
        
        # Check for specific optimization patterns
        if 'load' in original_ir and original_ir.count('load') > optimized_ir.count('load'):
            self.optimizations_applied.append("Load/store optimization applied")
        
        if 'br' in original_ir and original_ir.count('br') > optimized_ir.count('br'):
            self.optimizations_applied.append("Branch optimization applied")
        
        if 'call' in original_ir and original_ir.count('call') > optimized_ir.count('call'):
            self.optimizations_applied.append("Function inlining applied")


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize_code():
    """Main optimization endpoint"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        if not code.strip():
            return jsonify({
                'success': False,
                'error': 'No code provided'
            })
        
        if language.lower() == 'python':
            optimizer = PythonOptimizer()
        elif language.lower() == 'c':
            optimizer = COptimizer()
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported language: {language}'
            })
        
        result = optimizer.optimize(code)
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Optimization error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Check if required tools are available
    try:
        subprocess.run(['clang', '--version'], capture_output=True, check=True)
        subprocess.run(['opt', '--version'], capture_output=True, check=True)
        print("✓ LLVM tools available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Warning: LLVM tools (clang, opt) not found. C optimization will not work.")
        print("Install LLVM tools: sudo apt-get install llvm clang (Ubuntu) or brew install llvm (macOS)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)