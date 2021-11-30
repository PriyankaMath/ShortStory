# Unsupervised Translation of Programming Languages

![image](https://user-images.githubusercontent.com/21059586/144098657-8a0e289c-ff53-4af6-8ba7-f2d81d86cb4c.png)

Medium article: https://medium.com/@priyanka.math/unsupervised-translation-of-programming-languages-9d538c64096f

SlideShare Link: https://www.slideshare.net/PriyaM781673/trans-coder

![image](https://user-images.githubusercontent.com/21059586/144098139-2de939b5-0c42-48b7-a75a-0040c1ebbafa.png)

A transcompiler is a system that converts source code from one high-level programming language (such as C++ or Python) to another. Transcompilers are primarily used for interoperability, and to port codebases written in an obsolete languages (e.g. COBOL, Python 2) to a modern one. They typically rely on handcrafted rewrite rules applied on source code Abstract Syntax Tree (AST) that often lack readability, fail to comply with the target language conventions, and require manual modifications in order to work properly. The overall translation process is time-consuming and requires expertise in both the source and target languages. Although neural models significantly outperform their rule-based counterparts in the context of natural language translation, their applications to transcompilation have been limited due to the scarcity of parallel data in this domain. In this article, we will leverage recent approaches in unsupervised machine translation to train a fully unsupervised neural transcompiler. 
‘TransCoder’ model is trained on source code from open source GitHub projects, and show that it can translate functions between C++, Java, and Python with high accuracy. This approach relies exclusively on monolingual source code, requires no expertise in the source or target languages, and can easily be generalized to other programming languages. To evaluate the model, a test set of 852 parallel functions, along with associated unit tests are used to check the correctness of translations and show that the model outperforms rule-based approaches by a significant margin. 
For ‘TransCoder’, we consider a sequence-to-sequence (seq2seq) model with attention [2], composed of an encoder and a decoder with a transformer architecture [3]. We use a single shared model for all programming languages. 

‘TransCoder’ is trained using three principles of unsupervised machine translation, 
1.	Initialization
2.	Language modeling
3.	Back-translation

Let’s summarize these principles and learn how we instantiate them to translate programming languages.

1.	Cross Programming Language Model pretraining (Initialization)

The first principle initializes the model with cross-lingual masked language model pretraining. As a result, pieces of code that express the same instructions are mapped to the same representation, regardless of the programming language.

![image](https://user-images.githubusercontent.com/21059586/144099045-53de937c-0027-4456-84c6-57fc56319313.png)

Pretraining is a key ingredient of unsupervised machine translation. It ensures that sequences with a similar meaning are mapped to the same latent representation, regardless of their languages. Originally, pretraining was done by initializing the model with cross-lingual word representations. Subsequent work showed that pretraining the entire model (and not only word representations) in a cross-lingual way could lead to significant improvements in unsupervised machine translation. 
In particular, ‘TransCoder’ follows the pretraining strategy where a Cross-lingual Language Model (XLM) is pretrained with a masked language modeling objective on monolingual source code datasets.
The cross-lingual nature of the resulting model comes from the significant number of common tokens (anchor points) that exist across languages. In the context of English-French translation, the anchor points consists essentially of digits and city and people names. In programming languages, these anchor points come from common keywords (e.g. for, while, if, try), and also digits, mathematical operators, and English strings that appear in the source code.
For the masked language modeling (MLM) objective, at each iteration we consider an input stream of source code sequences, randomly mask out some of the tokens, and train ‘TransCoder’ to predict the tokens that have been masked out based on their contexts. We alternate between streams of batches of different languages. This allows the model to create high quality, cross-lingual sequence representations.

2.	Denoising auto-encoding (Language modeling)

Denoising auto-encoding, the second principle, trains the decoder to always generate valid sequences, even when fed with noisy data, and increases the encoder robustness to input noise.

![image](https://user-images.githubusercontent.com/21059586/144099232-5749a5d4-9ee7-466a-81cf-87c82bfb4d52.png)

The encoder and decoder of the seq2seq model is initialized with the XLM model pretrained in the previous step. 
XLM pretraining allows the seq2seq model to generate high quality representations of input sequences. However, the decoder lacks the capacity to translate, as it has never been trained to decode a sequence based on a source representation. To address this issue, we train the model to encode and decode sequences with a Denoising Auto-Encoding (DAE) objective. The DAE objective operates like a supervised machine translation algorithm, where the model is trained to predict a sequence of tokens given a corrupted version of that sequence.
The first symbol given as input to the decoder is a special token indicating the output programming language. At test time, a Python sequence can be encoded by the model, and decoded using the C++ start symbol to generate a C++ translation. The quality of the C++ translation will depend on the “cross-linguality” of the model: if the Python function and a valid C++ translation are mapped to the same latent representation by the encoder, the decoder will successfully generate this C++ translation.
The DAE objective also trains the “language modeling” aspect of the model, i.e. the decoder is always trained to generate a valid function, even when the encoder output is noisy. Moreover it also trains the encoder to be robust to input noise, which is helpful in the context of back-translation where the model is trained with noisy input sequences.

3.	Back-translation

Back-translation, the last principle, allows the model to generate parallel data which can be used for training. Whenever the Python → C++ model becomes better, it generates more accurate data for the C++ → Python model, and vice versa.

![image](https://user-images.githubusercontent.com/21059586/144099326-c0851878-6966-4d80-9f58-9621e4ce47ef.png)

In practice, XLM pretraining and denoising auto-encoding alone are enough to generate translations. However, the quality of these translations tends to be low, as the model is never trained to do what it is expected to do at test time, i.e. to translate functions from one language to another. To address this issue, we use back-translation(feedback), which is one of the most effective methods to leverage monolingual data in a weakly(semi)-supervised scenario. Initially, it was introduced to improve the performance of machine translation in the supervised setting, back-translation turned out to be an important component of unsupervised machine translation.
In the unsupervised setting, a forward source-to-target model is coupled with a backward target-to-source model trained in parallel. The target-to-source model is used to translate target sequences into the source language, producing noisy source sequences corresponding to the ground truth target sequences. The source-to-target model is then trained in a weakly supervised manner to reconstruct the target sequences from the noisy source sequences generated by the target-to-source model, and vice versa. The two models are trained in parallel until convergence.

Experiments

Training data:
GitHub public dataset available on Google BigQuery is used as dataset. Projects whose license explicitly permits the re-distribution of parts of the project are filtered and the C++, Java, and Python files within those projects are selected. Ideally, a transcompiler should be able to translate whole projects. But, ‘TransCoder’ translates at function level. Unlike files or classes, functions are short enough to fit into a single batch, and working at function level allows for a simpler evaluation of the model with unit tests. ‘TransCoder’ pretrains on all source code available, and train the denoising auto-encoding and back-translation objectives on functions only. It was observed that keeping comments in the source code increases the number of anchor points across languages, which results in a better overall performance. Therefore, comments were kept in the final datasets and experiments.

Preprocessing:
Recent approaches in multilingual natural language processing tend to use a common tokenizer, and a shared vocabulary for all languages. This reduces the overall vocabulary size, and maximizes the token overlap between languages, improving the cross-linguality of the model. In our case, a universal tokenizer would be suboptimal, as different languages use different patterns and keywords. The logical operators && and || exist in C++ where they should be tokenized as a single token, but not in Python. The indentations are critical in Python as they define the code structure, but have no meaning in languages like C++ or Java. We use the javalang5 tokenizer for Java, the tokenizer of the standard library for Python6, and the clang7 tokenizer for C++. These tokenizers ensure that meaningless modifications in the code (e.g. adding extra new lines or spaces) do not have any impact on the tokenized sequence.

![image](https://user-images.githubusercontent.com/21059586/144099608-04b0f773-8be1-48d2-a920-fc348cdf8bfe.png)

Evaluation:
GeeksforGeeks is an online platform with computer science and programming articles. It gathers many coding problems and presents solutions in several programming languages. From these solutions, set of parallel functions in C++, Java, and Python are extracted, to create validation and test sets. These functions not only return the same output, but also compute the result with similar algorithm.

![image](https://user-images.githubusercontent.com/21059586/144099685-02dcf32d-2a2c-4a15-8164-a31c2cff978c.png)

Analysis of results:

•	TransCoder successfully understands the syntax specific to each language, learns data structures and their methods, and correctly aligns libraries across programming languages. For instance, it learns to translate the ternary operator “X ? A : B” in C++ or Java to “if X then A else B” in Python, in an unsupervised way. 

•	TransCoder successfully map tokens with similar meaning to the same latent representation, regardless of their languages.

•	TransCoder can adapt to small modifications.

Example of unsupervised translation from Python to C++:
![image](https://user-images.githubusercontent.com/21059586/144099798-c2ded334-837f-4a33-99ce-3c2ab370ea3b.png)

Example of failed ‘TransCoder’ translations:

![image](https://user-images.githubusercontent.com/21059586/144099867-4c1bfc7b-18e9-49ab-9b27-2500f90b2b7b.png)

References:
[1] Marie-Anne Lachaux, Baptiste Roziere, Lowik Chanussot and Guillaume Lample. Unsupervised Translation of Programming Languages. arXiv preprint arXiv:2006.03511, 2020
[2] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pages 3104–3112, 2014
[3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.
