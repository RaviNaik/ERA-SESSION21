# ERA-SESSION21 - GPT from scratch
🤗[**Space Link**](https://huggingface.co/spaces/RaviNaik/ERA-SESSION21) 

This is an implementation of GPT [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) by Andrej Karpathy.  
Dataset used to train: [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

### Tasks:
1. :heavy_check_mark: Go through [this](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) video and train your GPT from scratch.
2. :heavy_check_mark: Upload the model to HuggingFace Apps, and share the link.

### Results
**Bigram Base model training and results**

![image](https://github.com/RaviNaik/ERA-SESSION21/assets/23289802/4cc02d93-98fc-4114-a4c9-8a3c249eaad3)

**GPT Model training results**

![image](https://github.com/RaviNaik/ERA-SESSION21/assets/23289802/95dcde00-bf20-4853-ad20-fa67c1046f6b)

#### Generation Output:
```python
model = torch.load("checkpoints/model.pth", map_location={"cpu": device})
results = generate("hello", model, block_size, 1000, device)
print(results)
```
```
hellows thence grown from thee.
Since thou hast raim, thou thast well were quarterned; and
ever man tree can saw for words word from her at hour
Whiles contrations or devoided from ere years;
Yea, foul vice, indelice on the bird of the
noble of Hermione.

PARIS:
Sir, adies, sir, hate no choping but to your good.

HENRY BOLINGBROKE:
Yes, to ask you might, foreweed.

WARCK:
'Tis he made moust true.

RORSET:
It is an hour fastal that cracknaf at the chase
Upon; you are your hearing news a daughter.

KING EDWARD IV:
Tut, Lord Warwick, thou shouldst aft Rutlansps?
Thou tust but back hild, he countemn'd my lady's seal,
For access dead the treature moon! and the Englisting!
Thy vage for yonder see thou be donen?
O, count thou dost not Romeo, thou pratheeo sir,
That sweet thou feigh with no past blood on
Be see, here through on that find bears, if an
pretterinctors three and aspect die meeds thou,
Behing mine of thy denigning state lain business?

SAMPSA:
Sir, ha! but thou refused? thyself food, gr
```
### Gradio Interface
![image](https://github.com/RaviNaik/ERA-SESSION21/assets/23289802/f339ec6b-17b3-4de6-bbef-14eb2b3fac84)

