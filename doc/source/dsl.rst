=========
Quant DSL
=========

    This article describes Quant DSL, a domain specific language for
    quantitative analytics. Language elements are combined according to
    the language syntax to form a statement of value which is evaluated
    according to the language semantics. The syntax of the language is
    defined with Backus–Naur Form. The semantics are defined with
    mathematical expressions commonly used within quantitative
    analytics. The validity of Monte Carlo simulation for all possible
    expressions in the language is proven by induction. Quant DSL has
    been implemented in Python as a part of the Quant software
    application.


Synthesis
=========

Quant DSL is a domain specific language for quantitative analytics.
The value of any domain specific language consists in obtaining a
declarative syntax by which domain functionality can be invoked
with potentially infinite variation, so that a complex domain can
be supported with relatively simple software. Once underlying
functionality has been abstracted to the level of a domain specific
language, support for a new case can be established in a relatively
short time. Because new code does not need to be written for a new
case, proliferation of code that is hard to test (and therefore
expensive to maintain) can be avoided.

Quant DSL is used to record and evaluate statements of value. It is
hoped that the elements of the Quant DSL (for example "Market",
"Fixing", "Settlement") are recognised as common terms within
quantitative analytics; whilst the elements are defined fully in
this article, each element may be familiar to people who are
familiar with the domain. Consequently, it is hoped that statements
of value that are written in Quant DSL will be readable (as well as
writable).

Given the infinite variation of expression in the language, it is
necessary to obtain an inductive proof of the integrity of the
language for any possible expression. Although alternative proofs
may be obtained, an inductive proof has been devised.

Quant DSL was invented during the development of the Quant Python
package. Quant is an open source application of both the SciPy
Python package (a library for scientific computation) and the
Domain Model Python package (a toolkit for enterprise
applications). Recent advances in software engineering practice
(for example Martin Fowler's patterns of enterprise application
architecture, the agile approach, or open source software) have
suggested new ways to obtain appropriate functionality. In
particular, the recent maturation of dynamic languages such as
Python means the focus of development can remain on the supported
domain. Quant has benefited from these tendencies.

Quant DSL is based on professional experience in financial
institutions in London, academic training in mathematics and
mathematical engineering, and professional experience in the
architecture and development of enterprise applications. It is
hoped that Quant DSL constitutes a fresh approach to a common
concern.

Download Full PDF
=================

Please download the `full PDF here <http://appropriatesoftware.net/quant/docs/quant-dsl-definition-and-proof.pdf>`_.

