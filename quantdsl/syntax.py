import ast
import importlib

import six

from quantdsl.exceptions import DslSyntaxError
from quantdsl.semantics import FunctionDef, DslNamespace


class DslParser(object):

    def __init__(self, dsl_classes=None):
        if dsl_classes is None:
            dsl_classes = {}
        self.dsl_classes = dsl_classes

    def parse(self, dsl_source, filename='<unknown>'):
        """
        Creates a DSL Module object from a DSL source text.
        """
        if not isinstance(dsl_source, six.string_types):
            raise DslSyntaxError("Can't dsl_parse non-string object", dsl_source)

        # assert isinstance(dsl_source, six.string_types)
        try:
            # Parse as Python source code, into a Python abstract syntax tree.
            ast_module = ast.parse(dsl_source, filename=filename, mode='exec')
        except SyntaxError as e:
            raise DslSyntaxError("DSL source code is not valid Python code: {}".format(dsl_source), e)

        # Generate Quant DSL from Python AST.
        return self.visitAstNode(ast_module)

    def visitAstNode(self, node):
        """
        Identifies which "visit" method to call, according to type of node being visited.

        Returns the result of calling the identified "visit" method.
        """
        # assert isinstance(node, ast.AST)

        # Construct the "visit" method name.
        dsl_element_name = node.__class__.__name__
        method_name = 'visit' + dsl_element_name

        # Try to get_quantdsl_app the "visit" method object.
        try:
            method = getattr(self, method_name)
        except AttributeError:
            msg = "element '%s' is not supported (visit method '%s' not found on parser): %s" % (
                dsl_element_name, method_name, node)
            raise DslSyntaxError(msg)

        # Call the "visit" method object, and return the result of visiting the node.
        return method(node=node)

    def visitModule(self, node):
        """
        Visitor method for ast.Module nodes.

        Returns a DSL Module, with a list of DSL expressions as the body.
        """
        # assert isinstance(node, ast.Module)
        body = []

        # Namespace for function defs in module.
        module_namespace = DslNamespace()

        for n in node.body:
            dsl_object = self.visitAstNode(n)

            if isinstance(dsl_object, FunctionDef):
                # Put function def in module namespace.
                module_namespace[dsl_object.name] = dsl_object
                # Share module namespace with this function.
                if dsl_object.module_namespace is None:
                    dsl_object.module_namespace = module_namespace

            # Include imported things.
            if isinstance(dsl_object, list):
                for _dsl_object in dsl_object:
                    if isinstance(_dsl_object, FunctionDef):
                        module_namespace[_dsl_object.name] = _dsl_object
            else:
                body.append(dsl_object)

        return self.dsl_classes['Module'](body, module_namespace, node=node)

    def visitImportFrom(self, node):
        """
        Visitor method for ast.ImportFrom nodes.

        Returns the result of visiting the expression held by the return statement.
        """
        assert isinstance(node, ast.ImportFrom)
        if node.module == 'quantdsl.semantics':
            return []
        from_names = [a.name for a in node.names]
        dsl_module = self.import_python_module(node.module)
        nodes = []
        for node in dsl_module.body:
            if isinstance(node, FunctionDef) and node.name in from_names:
                nodes.append(node)
        return nodes

    def import_python_module(self, module_name):
        nodes = []
        module = importlib.import_module(module_name)
        path = module.__file__.strip('c')
        source = open(path).read()  # .py not .pyc
        dsl_node = self.parse(source, filename=path)
        assert isinstance(dsl_node, self.dsl_classes['Module']), type(dsl_node)
        return dsl_node

    def visitReturn(self, node):
        """
        Visitor method for ast.Return nodes.

        Returns the result of visiting the expression held by the return statement.
        """
        # assert isinstance(node, ast.Return)
        return self.visitAstNode(node.value)

    def visitExpr(self, node):
        """
        Visitor method for ast.Expr nodes.

        Returns the result of visiting the contents of the expression node.
        """
        # assert isinstance(node, ast.Expr)
        assert isinstance(node.value, ast.AST), type(node.value)
        return self.visitAstNode(node.value)

    def visitNum(self, node):
        """
        Visitor method for ast.Name.

        Returns a DSL Number object, with the number value.
        """
        # assert isinstance(node, ast.Num)
        return self.dsl_classes['Number'](node.n, node=node)

    def visitStr(self, node):
        """
        Visitor method for ast.Str.

        Returns a DSL String object, with the string value.
        """
        # assert isinstance(node, ast.Str)
        return self.dsl_classes['String'](node.s, node=node)

    def visitUnaryOp(self, node):
        """
        Visitor method for ast.UnaryOp.

        Returns a specific DSL UnaryOp object (e.g UnarySub), along with the operand.
        """
        # assert isinstance(node, ast.UnaryOp)
        args = [self.visitAstNode(node.operand)]
        if isinstance(node.op, ast.USub):
            dsl_class = self.dsl_classes['UnarySub']
        else:
            raise DslSyntaxError("Unsupported unary operator token: %s" % node.op)
        return dsl_class(node=node, *args)

    def visitBinOp(self, node):
        """
        Visitor method for ast.BinOp.

        Returns a specific DSL BinOp object (e.g Add), along with the left and right operands.
        """
        # assert isinstance(node, ast.BinOp)
        type_map = {
            ast.Add: self.dsl_classes['Add'],
            ast.Sub: self.dsl_classes['Sub'],
            ast.Mult: self.dsl_classes['Mult'],
            ast.Div: self.dsl_classes['Div'],
            ast.Pow: self.dsl_classes['Pow'],
            ast.Mod: self.dsl_classes['Mod'],
            ast.FloorDiv: self.dsl_classes['FloorDiv'],
        }
        try:
            dsl_class = type_map[type(node.op)]
        except KeyError:
            raise DslSyntaxError("Unsupported binary operator token", node.op, node=node)
        args = [self.visitAstNode(node.left), self.visitAstNode(node.right)]
        return dsl_class(node=node, *args)

    def visitBoolOp(self, node):
        """
        Visitor method for ast.BoolOp.

        Returns a specific DSL BoolOp object (e.g And), along with the left and right operands.
        """
        # assert isinstance(node, ast.BoolOp)
        type_map = {
            ast.And: self.dsl_classes['And'],
            ast.Or: self.dsl_classes['Or'],
        }
        dsl_class = type_map[type(node.op)]
        values = [self.visitAstNode(v) for v in node.values]
        args = [values]
        return dsl_class(node=node, *args)

    def visitName(self, node):
        """
        Visitor method for ast.Name.

        Returns a DSL Name object, along with the name's string.
        """
        return self.dsl_classes['Name'](node.id, node=node)

    def visitCall(self, node):
        """
        Visitor method for ast.Call.

        Returns a built-in DSL expression, or a DSL FunctionCall if the name refers to a user
        defined function.
        """
        if node.keywords:
            raise DslSyntaxError("Calling with keywords is not currently supported (positional args only).")
        if node.starargs:
            raise DslSyntaxError("Calling with starargs is not currently supported (positional args only).")
        if node.kwargs:
            raise DslSyntaxError("Calling with kwargs is not currently supported (positional args only).")

        # Collect the call arg expressions (whose values will be passed into the call when it is made).
        call_arg_exprs = [self.visitAstNode(arg) for arg in node.args]

        # Check the called node is an ast.Name.
        called_node = node.func
        # assert isinstance(called_node, ast.Name)
        called_node_name = called_node.id

        # Construct a DSL object for this call.
        try:
            # Resolve the name with a new instance of a DSL class.
            dsl_class = self.dsl_classes[called_node_name]
        except KeyError:
            # Resolve as a FunctionCall, and expect
            # to resolve the name to a function def later.
            dsl_name_class = self.dsl_classes['Name']
            dsl_args = [dsl_name_class(called_node_name, node=called_node), call_arg_exprs]
            return self.dsl_classes['FunctionCall'](node=node, *dsl_args)
        else:
            dsl_object_class = self.dsl_classes['DslObject']
            assert issubclass(dsl_class, dsl_object_class), dsl_class
            return dsl_class(node=node, *call_arg_exprs)

    def visitFunctionDef(self, node):
        """
        Visitor method for ast.FunctionDef.

        Returns a named DSL FunctionDef, with a definition of the expected call argument values.
        """
        name = node.name
        dsl_function_arg_class = self.dsl_classes['FunctionArg']
        if six.PY2:
            arg_name_attr = 'id'
        else:
            arg_name_attr = 'arg'
        call_arg_defs = [dsl_function_arg_class(getattr(arg, arg_name_attr), '') for arg in node.args.args]
        assert len(node.body) == 1, "Function defs with more than one body statement are not supported at the moment."
        decorator_names = [ast_name.id for ast_name in node.decorator_list]
        body = self.visitAstNode(node.body[0])
        dsl_args = [name, call_arg_defs, body, decorator_names]
        function_def = self.dsl_classes['FunctionDef'](node=node, *dsl_args)
        return function_def

    def visitIfExp(self, node):
        """
        Visitor method for ast.IfExp.

        Returns a named DSL IfExp, with a test DSL expression and expressions whose usage is
        conditional upon the test.
        """
        test = self.visitAstNode(node.test)
        body = self.visitAstNode(node.body)
        orelse = self.visitAstNode(node.orelse)
        args = [test, body, orelse]
        return self.dsl_classes['IfExp'](node=node, *args)

    def visitIf(self, node):
        """
        Visitor method for ast.If.

        Returns a named DSL If object, with a test DSL expression and expressions whose usage is
        conditional upon the test.
        """
        test = self.visitAstNode(node.test)
        assert len(node.body) == 1, "If statements with more than one body statement are not supported at the moment."
        body = self.visitAstNode(node.body[0])
        assert len(
            node.orelse) == 1, "If statements with more than one orelse statement are not supported at the moment."
        orelse = self.visitAstNode(node.orelse[0])
        args = [test, body, orelse]
        return self.dsl_classes['If'](node=node, *args)

    def visitCompare(self, node):
        """
        Visitor method for ast.Compare.

        Returns a named DSL Compare object, with operators (ops) and operands (comparators).
        """

        left = self.visitAstNode(node.left)
        op_names = [o.__class__.__name__ for o in node.ops]
        comparators = [self.visitAstNode(c) for c in node.comparators]
        args = [left, op_names, comparators]
        return self.dsl_classes['Compare'](node=node, *args)
