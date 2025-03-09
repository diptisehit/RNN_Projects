# RNN

Recurrent Neural Networks (RNNs) work a bit different from regular neural networks. In neural network the information flows in one direction from input to output. However in RNN information is fed back into the system after each step.

RNNs allow the network to “remember” past information by feeding the output from one step into next step. This helps the network understand the context of what has already happened and make better predictions based on that. For example when predicting the next word in a sentence the RNN uses the previous words to help decide what word is most likely to come next.

# Recurrent Neural Network Architecture

![image](https://github.com/user-attachments/assets/a340f4de-8f36-4dd2-87b2-74abc35440f2)

Unlike traditional deep neural networks, where each dense layer has distinct weight matrices, RNNs use shared weights across time steps, allowing them to remember information over sequences.

In RNNs, the hidden state Hi is calculated for every input Xi to retain sequential dependencies. The computations follow these core formulas:

1. Hidden State Calculation:

ht = σ( Wx * X  +   Wh * ht−1 + B)

Here, ht represents the current hidden state, Wx and Wh are weight matrices and B is the bias.

![image](https://github.com/user-attachments/assets/83aa1b99-4c77-42f1-8f24-d24b8c16a509)

This image showcases the basic architecture of RNN and the feedback loop mechanism where the output is passed back as input for the next time step.

# How RNN Differs from Feedforward Neural Networks?

Feedforward Neural Networks (FNNs) process data in one direction from input to output without retaining information from previous inputs. This makes them suitable for tasks with independent inputs like image classification. However FNNs struggle with sequential data since they lack memory.

Recurrent Neural Networks (RNNs) solve this by incorporating loops that allow information from previous steps to be fed back into the network. This feedback enables RNNs to remember prior inputs making them ideal for tasks where context is important.

# How does RNN work?
At each time step RNNs process units with a fixed activation function. These units have an internal hidden state that acts as memory that retains information from previous time steps. This memory allows the network to store past knowledge and adapt based on new inputs.

#  Backpropagation through time (BPTT) 

Since RNNs process sequential data Backpropagation Through Time (BPTT) is used to update the network’s parameters. The loss function L(θ) depends on the final hidden state and each hidden state relies on preceding ones forming a sequential dependency chain.
BPTT is a learning process where errors are propagated across time steps to adjust the network’s weights enhancing the RNN’s ability to learn dependencies within sequential data.

# Types Of Recurrent Neural Networks

There are four types of RNNs based on the number of inputs and outputs in the network:

1. One-to-One RNN : single input and a single output e.g. binary classification where no sequential data is involved.
2. One-to-Many RNN :a single input used to produce multiple outputs over time. e.g. in image captioning a single image can be used as input to generate a sequence of words as a caption.
3. Many-to-One RNN : a sequence of inputs and generates a single output. e.g. sentiment analysis
4. Many-to-Many RNN : processes a sequence of inputs and generates a sequence of outputs. e.g. language translation

# Limitations of Recurrent Neural Networks (RNNs)
While RNNs excel at handling sequential data, they face two main training challenges :

1. Vanishing Gradient: During backpropagation, gradients diminish as they pass through each time step, leading to minimal weight updates. This limits the RNN’s ability to learn long-term dependencies, which is crucial for tasks like language translation.
2. Exploding Gradient: Sometimes, gradients grow uncontrollably, causing excessively large weight updates that destabilize training.

# Variants of Recurrent Neural Networks (RNNs)
1. Vanilla RNN
2. Bidirectional RNNs
3. Long Short-Term Memory Networks (LSTMs)
4. Gated Recurrent Units (GRUs)





