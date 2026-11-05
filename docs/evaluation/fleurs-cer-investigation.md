# FLEURS CER investigation

Per-utterance results for 834 FLEURS Slovene utterances.

- **ours**: 7424 character edits over 104299 reference chars = 7.12% CER
- **stock**: 6520 character edits over 104299 reference chars = 6.25% CER

## Is it a few utterances or many?

- worst **1** utterance(s) account for **2%** of our excess character edits
- worst **5** utterance(s) account for **10%** of our excess character edits
- worst **10** utterance(s) account for **18%** of our excess character edits
- worst **25** utterance(s) account for **36%** of our excess character edits
- worst **50** utterance(s) account for **57%** of our excess character edits

## Failure signatures

| signature | ours | stock |
|---|---|---|
| empty output | 0 | 0 |
| runaway (hyp > 2x ref chars) | 4 | 4 |
| truncated (hyp < 0.5x ref chars) | 2 | 2 |

## The five worst utterances for us

```
ref   ( 120 chars) zaradi tega bo povratno združljiv s standardi 802 11a 802 11b in 802 11g v kolikor bo imela osnovna postaja dvojni radio
ours  ( 172 chars) zaradi tega bo povratno združljiv standardi osemsto dva pika enajst a osemsto dva pika enajst b in osemsto dva pika enajst g v kolikor bo imela osnovn
stock ( 117 chars) zaradi tega bo povratno združljiv standardi 802 11a 802 11b in 802 11g vkolikor bo imela osnovna postaja dvojni radio
excess character edits vs stock: 68
```

```
ref   (  61 chars) standard 802 11n deluje tako na frekvenci 2 4 ghz kot 5 0 ghz
ours  ( 111 chars) standard osemsto dva enajst en deluje tako na frekvenci dva cela štiri gigahertzov kot pet cela nič gigahertzov
stock (  61 chars) standard 802 11n deluje tako na frekvenci 2 4 ghz kot 5 1 ghz
excess character edits vs stock: 58
```

```
ref   ( 120 chars) zaradi tega bo povratno združljiv s standardi 802 11a 802 11b in 802 11g v kolikor bo imela osnovna postaja dvojni radio
ours  ( 157 chars) zaradi tega bo povratno združil standardi osemsto dva enajst a osemsto dva enajst b in osemsto drugi enajst g v kolikor bo imela osnovna postaja dvojn
stock ( 117 chars) zaradi tega bo povratno združljiv standardi 802 11a 802 11b in 802 11g vkolikor bo imela osnovna postaja dvojni radio
excess character edits vs stock: 58
```

```
ref   ( 120 chars) zaradi tega bo povratno združljiv s standardi 802 11a 802 11b in 802 11g v kolikor bo imela osnovna postaja dvojni radio
ours  ( 162 chars) zaradi tega bo povratno združljiv standardi osemsto dva pika enajst a osemsto dva enajst b in osemsto dva enajst g v kolikor bo imela osnovna postaja 
stock ( 117 chars) zaradi tega bo povratno združljiv standardi 802 11a 802 11b in 802 11g vkolikor bo imela osnovna postaja dvojni radio
excess character edits vs stock: 58
```

```
ref   ( 143 chars) sundarbans je od leta 1966 rezervat za divje živali in po ocenah naj bi bilo danes na tem območju 400 bengalskih tigrov in okoli 30 000 čitalov
ours  ( 179 chars) sundbarks je od leta tisoč devetsto šestinšestdeset rezervac za delo živali in po ocenah naj bi bilo danes na temo močjo štiristo benganskih tigrov in
stock ( 143 chars) sundbarks je od leta 1966 rezervac za devje živali in po ocenah naj bi bilo danes na temo močjo 400 benganskih tigrov in okoli 30 tisoč čitalov
excess character edits vs stock: 49
```

## Decode-setting sweep

Whisper's runaway and early-stop behaviour is governed at decode time.
Each row is the full 834-utterance set, our model, same normalisation.

| decode setting | WER | CER | empty | runaway |
|---|---|---|---|---|
| baseline (as shipped) | 14.53% | 7.12% | 0 | 4 |
| condition_on_previous_text=False | 14.53% | 7.12% | 0 | 4 |
| vad_filter=True | 14.66% | 7.13% | 0 | 4 |
| both | 14.66% | 7.13% | 0 | 4 |

**Best: baseline (as shipped)** — WER 14.53%, CER 7.12%

Baseline was WER 14.53% / CER 7.12%. Stock whisper-large-v3 on the same
set: WER 20.92%% / CER 6.25%%.

Raw sweep data: /tmp/cer_sweep.json
