# SoccerNet Dense Video Captioning チャレンジ

video_blip をファインチューニングしてみたい。

## なぜやるのか

- 来年の SoccerNet チャレンジの予行演習
- 実装力の向上 & 分散学習や重いデータセットの扱いに慣れる
- 来年以降の修士研究の糧にする

## どうやるのか

**重要: 手戻りが発生しないように、丁寧にタスクを洗い出す。このステップをサボらない。**

1. 事前調査

   - [ ] 利用されているデータセットの形式(m3it, llava のような csv 形式のデータセットに変換する。)
     - [x] どうやら、image & prompt(instruction & prompt の場合もある) → caption の形式になっているようだが、入力 prompt をどうするか？→ 固定の instruction を入力する
     - [x] 画像が image_base64_str になってる。簡単に扱えそうで嬉しい。ただ、メモリに base64 文字列を載せ切れる気がしない。imag_path を載せる方式にしようかしら？
     - [x] imag_path にするとして、画像を書き出す必要がある
     - num_frame を合わせる必要があるかも？
     - [x] "###human {}, ###gpt {}" という形式で、instruction tuning っぽくなってる。この形式はこのまま利用する？→ そのまま
   - [ ] config ファイルの書き方
   - [ ] 学習方法（notebook 見るだけで良い）
   - [ ] 事前学習済みのものがないか調べる。(BLIP, BLIP2 を ActivityNet や Kinetics で事前学習したものがあるかもしれない。できればそれ使いたい。なぜなら SoccerNet のデータ量は限られているから、事前学習なしだとうまくいかないと思うから)
   - [ ] pretrain model が見つかったら、https://github.com/huggingface/tokenizers/issues/247 を参考に、tokenizer の vocab を拡張する。

   ```python
   special_tokens_dict = {'additional_special_tokens': ['[C1]','[C2]','[C3]','[C4]']}
   num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
   model.resize_token_embeddings(len(tokenizer))
   ```

2. データセットの準備

   - [ ] 現状の SoccerNet 形式の調査(SwinBERT で使えるように変換した際に、既にうまくできるようになったら嬉しい..?)
   - [ ] 差分をどう埋めるか考える
   - [ ] データセットの変換

3. 実際の学習に必要な追加すべきファイルを作成する

   - [ ] datasets/soccernet_datasets.py
   - [ ] configs/datasets/soccernet.yaml
   - [ ] projects/video_blip/simple.yml
   - [ ] なんらかの評価モジュール (trainer に compute_metrics Callback 引数に渡せる形にする。)
     - [ ] blue, rouge, cider, meteor などの実装？

4. 学習を実行する

   - [ ] コード作成 notebooks/finetune_soccernet.ipynb
   - [ ] 実行 (おそらく gpu20 でやる)

```

```
