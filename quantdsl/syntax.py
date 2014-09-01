import ast

from quantdsl.exceptions import DslSyntaxError


class DslParser(object):

    def parse(self, dslSource, filename='<unknown>', dslClasses=None):
        """
        Creates a DSL Module object from a DSL source text.
        """
        self.dslClasses = {}
        if dslClasses:
            assert isinstance(dslClasses, dict)
            self.dslClasses.update(dslClasses)

        if not isinstance(dslSource, basestring):
            raise DslSyntaxError("Can't parse non-string object", dslSource)

        assert isinstance(dslSource, basestring)
        try:
            # Parse as Python source code, into a Python abstract syntax tree.
            astModule = ast.parse(dslSource, filename=filename, mode='exec')
        except SyntaxError, e:
            raise DslSyntaxError("DSL source code is not valid Python code", e)

        # Generate Quant DSL from Python AST.
        return self.visitAstNode(astModule)

    def visitAstNode(self, node):
        """
        Identifies which "visit" method to call, according to type of node being visited.

        Returns the result of calling the identified "visit" method.
        """
        assert isinstance(node, ast.AST)

        # Construct the "visit" method name.
        dslElementName = node.__class__.__name__
        methodName = 'visit' + dslElementName

        # Try to get the "visit" method object.
        try:
            method = getattr(self, methodName)
        except AttributeError:
            msg = "element '%s' is not supported (visit method '%s' not found on parser): %s" % (
                dslElementName, methodName, node)
            raise DslSyntaxError(msg)

        # Call the "visit" method object, and return the result of visiting the node.
        return method(node=node)

    def visitReturn(self, node):
        """
        Visitor method for ast.Return nodes.

        Returns the result of visiting the expression held by the return statement.
        """
        assert isinstance(node, ast.Return)
        return self.visitAstNode(node.value)

    def visitModule(self, node):
        """
        Visitor method for ast.Module nodes.

        Returns a DSL Module, with a list of DSL expressions as the body.
        """
        assert isinstance(node, ast.Module)
        body = [self.visitAstNode(n) for n in node.body]
        return self.dslClasses['Module'](body, node=node)

    def visitExpr(self, node):
        """
        Visitor method for ast.Expr nodes.

        Returns the result of visiting the contents of the expression node.
        """
        assert isinstance(node, ast.Expr)
        if isinstance(node.value, ast.AST):
            return self.visitAstNode(node.value)
        else:
            raise DslSyntaxError

    def visitNum(self, node):
        """
        Visitor method for ast.Name.

        Returns a DSL Number object, with the number value.
        """
        assert isinstance(node, ast.Num)
        return self.dslClasses['Number'](node.n, node=node)

    def visitStr(self, node):
        """
        Visitor method for ast.Str.

        Returns a DSL String object, with the string value.
        """
        assert isinstance(node, ast.Str)
        return self.dslClasses['String'](node.s, node=node)

    def visitUnaryOp(self, node):
        """
        Visitor method for ast.UnaryOp.

        Returns a specific DSL UnaryOp object (e.g UnarySub), along with the operand.
        """
        assert isinstance(node, ast.UnaryOp)
        args = [self.visitAstNode(node.operand)]
        if isinstance(node.op, ast.USub):
            dslUnaryOpClass = self.dslClasses['UnarySub']
        else:
            raise DslSyntaxError("Unsupported unary operator token: %s" % node.op)
        return dslUnaryOpClass(node=node, *args)

    def visitBinOp(self, node):
        """
        Visitor method for ast.BinOp.

        Returns a specific DSL BinOp object (e.g Add), along with the left and right operands.
        """
        assert isinstance(node, ast.BinOp)
        typeMap = {
            ast.Add: self.dslClasses['Add'],
            ast.Sub: self.dslClasses['Sub'],
            ast.Mult: self.dslClasses['Mult'],
            ast.Div: self.dslClasses['Div'],
            ast.Pow: self.dslClasses['Pow'],
            ast.Mod: self.dslClasses['Mod'],
            ast.FloorDiv: self.dslClasses['FloorDiv'],
        }
        try:
            dslClass = typeMap[type(node.op)]
        except KeyError:
            raise DslSyntaxError("Unsupported binary operator token", node.op, node=node)
        args = [self.visitAstNode(node.left), self.visitAstNode(node.right)]
        return dslClass(node=node, *args)

    def visitBoolOp(self, node):
        """
        Visitor method for ast.BoolOp.

        Returns a specific DSL BoolOp object (e.g And), along with the left and right operands.
        """
        assert isinstance(node, ast.BoolOp)
        typeMap = {
            ast.And: self.dslClasses['And'],
            ast.Or: self.dslClasses['Or'],
        }
        try:
            dslClass = typeMap[type(node.op)]
        except KeyError:
            raise DslSyntaxError("Unsupported boolean operator token: %s" % node.op)
        else:
            values = [self.visitAstNode(v) for v in node.values]
            args = [values]
            return dslClass(node=node, *args)

    def visitName(self, node):
        """
        Visitor method for ast.Name.

        Returns a DSL Name object, along with the name's string.
        """
        return self.dslClasses['Name'](node.id, node=node)

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
        callArgExprs = [self.visitAstNode(arg) for arg in node.args]

        # Check the called node is an ast.Name.
        calledNode = node.func
        assert isinstance(calledNode, ast.Name)
        calledNodeName = calledNode.id

        # Construct a DSL object for this call.
        try:
            # Resolve the name with a new instance of a DSL class.
            dslClass = self.dslClasses[calledNodeName]
        except KeyError:
            # Resolve as a FunctionCall, and expect
            # to resolve the name to a function def later.
            dslNameClass = self.dslClasses['Name']
            dslArgs = [dslNameClass(calledNodeName, node=calledNode), callArgExprs]
            return self.dslClasses['FunctionCall'](node=node, *dslArgs)
        else:
            baseDslObjectClass = self.dslClasses['DslObject']
            assert issubclass(dslClass, baseDslObjectClass), dslClass
            return dslClass(node=node, *callArgExprs)

    def visitFunctionDef(self, node):
        """
        Visitor method for ast.FunctionDef.

        Returns a named DSL FunctionDef, with a definition of the expected call argument values.
        """
        name = node.name
        dslFunctionArgClass = self.dslClasses['FunctionArg']
        callArgDefs = [dslFunctionArgClass(arg.id, '') for arg in node.args.args]
        assert len(node.body) == 1, "Function defs with more than one body statement are not supported at the moment."
        decoratorNames = [astName.id for astName in node.decorator_list]
        body = self.visitAstNode(node.body[0])
        dslArgs = [name, callArgDefs, body, decoratorNames]
        functionDef = self.dslClasses['FunctionDef'](node=node, *dslArgs)
        return functionDef

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
        return self.dslClasses['IfExp'](node=node, *args)

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
        return self.dslClasses['If'](node=node, *args)

    def visitCompare(self, node):
        """
        Visitor method for ast.Compare.

        Returns a named DSL Compare object, with operators (ops) and operands (comparators).
        """

        left = self.visitAstNode(node.left)
        opNames = [o.__class__.__name__ for o in node.ops]
        comparators = [self.visitAstNode(c) for c in node.comparators]
        args = [left, opNames, comparators]
        return self.dslClasses['Compare'](node=node, *args)
