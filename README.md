# GPT-2 demó

Ez egy demo alkalmazás, amely az OpenAI GPT-2 mesterséges intelligenciás nyelvi model tanulási célú, Java nyelvű megvalósítása.

Célom a Transformer architektúra megértése és bemutatása volt, nem egy használható, optimalizált program elkészítése. Éppen ezért nem használtam TensorFlow-t vagy más neurális hálózatok megvalósítására szolgáló utility-t, hanem minden egyes műveletet magam programoztam le. Az eredmény azonban olyan lassú lett, hogy végül az alapműveletek végrehajtására mégis igénybe vettem az Nd4j csomagot, azonban ennek használata opcionális, és mivel a naív megvalósításom is ott van mellette, egyértelműen látszik, hogy milyen lépések kerülnek végrehajtásra.

(A programot részletesen kommenteltem, és igyekeztem olyan egyszerű kódot írni, hogy akár nem Java programozók is könnyen olvasni tudják.)

## Használat ##

Lépések:

1. Telepítsük fel a Java fejlesztői csomagot, valamint a Maven build tool-t, és opcionálisan a Git verziókezelőt!


2. Töltsük le ezt a forráskódot, vagy a Git használata esetén "klónozzuk a repót": https://github.com/huplay/gpt2-demo

    ```git clone https://github.com/huplay/gpt2-demo.git```


3. Töltsük le a betanítás során előállt paraméterfájlokat. (Elég csak azt, amelyiket használni szeretnénk.) Az alapértelmezett útvonal a gpt2-demo/parameters folder, tehát a legegyszerűbb, ha ide másoljuk át a fájlokat, de a programnak más útvonal is konfigurálható. Itt találhatók a paraméterek: 
- SMALL: https://github.com/huplay/gpt2-demo-small-params
- MEDIUM: https://github.com/huplay/gpt2-demo-medium-params
- LARGE: https://github.com/huplay/gpt2-demo-large-params
- XL: https://github.com/huplay/gpt2-demo-xl-params1,
      
   https://github.com/huplay/gpt2-demo-xl-params2

   
4. Sajnos a GitHub-on van egy maximum fájlméret korlát, ezért a legnagyobb paraméterfájlokat nem lehet egyben feltölteni. (Ez minden méret esetén csupán a wte.dat fájl-t jelenti.) Ezért ezt több részre szabdalva töltöttem fel, de használat előtt egyesíteni kell ezeket (   ```wte.001```, ```wte.002```...)

   Az egyesítés megoldható a Total Commander "combine files" (fájlegyesítés?) funkciójával, vagy a cmd-n belül az alábbi paranccsal: ```copy /B wte.001 + wte.002 wte.dat``` (A nagyobb változatok esetén + wte.003 vagy + wte.004 is szükséges.)


5. Parancssor program (`cmd`) használatával lépjünk be a GPT2-demo főkönyvtárába:
   
    ```cd gpt2-demo```


6. Fordítsuk le a kódot a Maven használatával (build):

    ```mvn clean install```


7. Indítsuk el a programot:

    Windows alatt: ```run.bat``` (SMALL verzió)
   
   (Vagy a nagyobb verziókhoz: ```runmedium.bat```, ```runlarge.bat```, ```runxl.bat```)
    
   Bármilyen rendszer esetén:```java -jar target/gpt2-demo-1.0-jar-with-dependencies.jar```

   
A program indulás után bekér egy kezdő szöveget, a feladat pedig az, hogy ezt folytassa:

```Input text:```

(Ha ki akarunk lépni, egyetlen `q` betűt írjunk csak be, + Enter.)

A válasz kiírása után a szöveg bekérése újra és újra megismétlődik, de minden alkalommal egy teljesen új menet kezdődik, tehát a válaszok során nem fog emlékezni a korábbi bemenetekre és válaszokra. (A GPT-2 nem csetelésre lett kitalálva.)

## A program paraméterei ##

- ``model`` - Model méret: SMALL (default), MEDIUM, LARGE, XL
- ``util`` - A használt utility: ND4J (default), STANDARD (amit én írtam)
- ``path`` - A paraméter fájlok útvonala (default: /parameters) 
- ``maxlength`` - Generált tokenek maximális száma.
- ``topk`` - Az eredmény kiválasztása során ennyi legvalószínűbb lehetőség közül választ véletlenszerűen a rendszer

## Betanítás ##

A program nem képes tanulni, az OpenAI által betanított GPT-2 tanítás során előállt paramétereit fájlokból kell betölteni. Ezeket az eredeti GPT-2 Python alkalmazásból nyertem ki, majd Java bináris formátumúvá alakítottam.

https://github.com/openai/gpt-2

## Történelem ##

### Transformer ###

- Attention Is All You Need (2017, Google Brain)
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Usykoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- https://arxiv.org/abs/1706.03762 
- https://arxiv.org/pdf/1706.03762.pdf

Ez a tanulmány írta le az enkóderek és dekóderek használatával működő Transformer architektúrát, elsősorban két nyelv közti fordítás esetére.
Az enkóderek az egyik nyelv szövege alapján előállítanak egy belső reprezentációt, melyet a dekóderek egy másik nyelvű szöveggé alakítanak.
(Az enkóderek által létrehozott key és value vektorokat használják a dekódere.
Az itt leírt rendszer 6 enkódert és 6 dekódert tartalmazott.

### Csak dekóderes transformer ###

- Generating Wikipedia by summarizing long sentences (2018, Google Brain)
- Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Łukasz Kaiser, Noam Shazeer
- https://arxiv.org/pdf/1801.10198.pdf

A Google csapata kicsit később leírt egy csak dekóderekből álló módosított Transformer architektúrát is, melyet természetes nyelvek modellezésére szántak.
Itt a dekóder maga állítja elő a query, key és value vektorokat is (hasonlóan, mint az enkóderek).

### GPT-1 ###

- Improving Language Understanding by Generative Pre-Training (2018, OpenAI)
- Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
- https://openai.com/blog/language-unsupervised/
- https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- Source code: https://github.com/openai/finetune-transformer-lm

Az Google által publikált megoldások alapján az OpenAI elkészítette saját változatát. A legelső GPT 12 dekódert és 12 attention fejet (head) tartalmazott.
Annyit változtattak, hogy az eredeti szinuszos pozíció beágyazás helyett a pozíció reprezentációját a betanítás során kalkulálták ki.

### GPT-2 ###

- Language Models are Unsupervised Multitask Learners (2019, OpenAI)
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- https://openai.com/blog/better-language-models/
- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Source code: https://github.com/openai/gpt-2

A GPT-2-t négy féle méretben készítették el. A legkisebb (SMALL) a GPT-1-gyel megegyező méretű volt (12 dekóder, 12 head), a legnagyobb (XL) 48 dekódert és 25 head-et használt.

Az egyetlen architekturális változás a GPT-1-hez képest az, hogy a dekódereken belüli normalizáció az attention és feed forward lépések utánról azok előttre került.
(Tehát nem att/add/norm/mlp/add/norm a lépések sorrendje, hanem norm/att/add/norm/mlp/add.)
 
## Segítség (angolul) ##

Jay Alammar: 
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- The Illustrated GPT-2: https://jalammar.github.io/illustrated-gpt2

