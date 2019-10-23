# Syntax Tree-Network
Tree-structured neural network that uses explicit syntax as an inductive bias to learn compositional representations of language.

Remember to sum loss across batch dimension instead of average so that learning rate is effectively dynamic based on size of the current batch.