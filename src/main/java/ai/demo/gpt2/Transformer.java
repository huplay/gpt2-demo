package ai.demo.gpt2;

import ai.demo.gpt2.util.Util;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Csak dekóderekből álló Transformer megvalósítás, ugyanolyan, mint az OpenAI GPT-2
 */
public class Transformer
{
    private static final int END_OF_TEXT = 50256;
    private static final float EPSILON = 1e-5f;

    private final Config config;
    private final Parameters params;
    private final Util util;

    private final TransformerDecoder[] decoders;

    /**
     * Konstruktor, értékek inicializálása
     */
    public Transformer(Config config, Parameters params)
    {
        this.config = config;
        this.params = params;
        this.util = config.utilType.util;

        // Create the decoder stack
        this.decoders = new TransformerDecoder[config.modelType.decoderCount];
        for (int i = 0; i < config.modelType.decoderCount; i++)
        {
            this.decoders[i] = new TransformerDecoder(config, params.decoderParameters[i], EPSILON);
        }
    }

    /**
     * Transformer token-adagolási logika
     *
     * Ez a metódus adagolja be a bemenő tokeneket a Transformernek,
     * majd az utolsó bemenetre adott kimenetet (eredményt) újra és újra beadja bemenetként
     *
     * @param inputTokens A bemenő szöveg tokenjeinek listája
     * @return - Az összes eredményül kapott token listája
     */
    public List<Integer> processTokens(List<Integer> inputTokens)
    {
        // Előállított tokenek gyűjtője
        List<Integer> result = new ArrayList<>();

        // Szövegen belüli pozíció számlálója
        int pos = 0;

        if (inputTokens.size() == 0)
        {
            // Ha üres a bemenet, egy END_OF_TEXT tokent használunk helyette
            inputTokens.add(END_OF_TEXT);
        }
        else
        {
            // Végigmegyünk a bemenő tokeneken (leszámítva az utolsót), és beadjuk őket a Transformernek
            for (; pos < inputTokens.size() - 1; pos++)
            {
                // A Transformer kimenete most még nem érdekel minket, ezért nem olvassuk ki a visszatérési értéket,
                // de ezalatt a dekóderek eltárolják az adott tokenre vonatkozó adatokat (előállított key és value vektorokat)
                processToken(pos, inputTokens.get(pos));
            }
        }

        // Az utolsó bemenő token feldolgozása. A kimeneti érték lesz az első újonnan generált token
        float[] embedding = processToken(pos, inputTokens.get(pos));
        int nextToken = determineOutput(embedding);
        result.add(nextToken);

        // Ezután az újonnan kapott tokeneket újra és újra beadjuk a Transformernek, még újabb tokeneket generálva
        for (pos++; pos < config.maxLength + inputTokens.size(); pos++)
        {
            // Az előző menetben generált token feldolgozása
            embedding = processToken(pos, nextToken);

            // Az eredményül kapott lehetőségek közül kiválasztunk egyet
            nextToken = determineOutput(embedding);
            result.add(nextToken);

            // Kilépés, ha az eredmény az END_OF_TEXT token volt, vagy elértük a maximum tokenszámot
            if (nextToken == END_OF_TEXT || (inputTokens.size() + result.size() >= config.modelType.contextSize)) break;
        }

        return result;
    }

    /**
     * Tranformer logika - egyetlen token feldolgozása
     *
     * @param pos - Aktuális bemenő token pozíciója (hanyadik a szövegben)
     * @param token - Az aktuális bemenő token azonosítója
     * @return eredményül egy embedding vektort kapunk
     * (Közben a dekóderek eltárolják a key és value vektorokat, melyeket későbbi tokenek feldolgozása során használni fogunk.)
     */
    private float[] processToken(int pos, int token)
    {
        // Kikeressük a bemenő tokenhez tartozó embedding vektort - ezen zajlik majd a feldolgozás
        float[] embedding = params.tokenEmbeddings[token];

        // Hozzáadjuk a pozícióra jellemző információt
        embedding = util.addVectors(embedding, params.positionEmbeddings[pos]);

        // Dekóderek sorozata
        for (TransformerDecoder decoder : decoders)
        {
            embedding = decoder.calculate(embedding);
        }

        // Utolsó normalizáció
        embedding = util.normalize(embedding, EPSILON);
        for (int i = 0; i < embedding.length; i++)
        {
            embedding[i] = embedding[i] * params.normFinalWeights[i] + params.normFinalBiases[i];
        }

        return embedding;
    }

    /**
     * Meghatározzuk (kiválasztjuk) az eredmény tokent a kimeneti embedding alapján
     *
     * @param embedding - a feldolgozás során eredményül kapott embedding
     * @return a kiválasztott token azonosítója
     */
    private int determineOutput(float[] embedding)
    {
        // A tokenek valószínűségét* egy egyszerű mátrix-szorzással kaphatjuk meg.
        // Az itt bemenetül kapott embedding egy vektor (vagyis egysoros mátrix), tehát a dimenziói: 1 * embedding-hossz.
        // A betanítás során előállított word-token-embedding (wte) mátrix egy hatalmas táblázat,
        // amely minden tokenhez hozzárendel egy kiinduló embeddinget, vagyis ennek dimenziói: 50257 * embedding-hossz.
        // Az 1 * embedding-hossz mátrixot meg lehet szorozni egy embedding-hossz * 50257 méretűvel (a wte transzponáltjával),
        // az eredmény pedig egy 1 * 50257 méretű mátrix lesz, vagyis 50257 darab szám,
        // amely token-valószínűségekként* értelmezendő, a token azonosítók szerinti sorrendben.

        // * Valójában ez nem valószínűség, de egy ahhoz hasonló dolog, amit logit-nak neveznek. logit = ln(valószínűség / 1 - valószínűség)
        //   A valószínűség egy 0 és 1 közti érték, míg a logit lehet negatív is, és tetszőleges méretű,
        //   de a kisebb logit kisebb valószínűséget jelent, ezért a logit-ok végzett szűrés (legnagyobb értékek megtartása)
        //   valószínűségeken végzett szűrésre is jó.
        float[] logits = util.multiplyVectorByTransposedMatrix(embedding, params.tokenEmbeddings);

        // Itt meg lehetne valósítani a temperature és topP alapú szűrést is, de most csak a topK-t csináltam meg:
        // temperature logikája: el kell osztani a logit értékeket a temperature értékével (0 és 1 közti számmal)
        // topP szűrés logikája: a valószínűség szerint csökkenő sorrendbe állított lehetőségek közül azokat kell meghagyni,
        // melyek valószínűsége együtt elér egy bizonyos szintet (a többit elvetjük)

        // topK szűrés

        // A logitok listáját egy két oszlopból álló mátrix-szá alakítjuk:
        // Az első oszlop a logit, a második pedig a token azonosító (hanyadik token)
        float[][] indexedLogits = new float[logits.length][2];
        for (int i = 0; i < logits.length; i++)
        {
            indexedLogits[i][0] = logits[i];
            indexedLogits[i][1] = i;
        }

        // Az első oszlop (logit) alapján csökkenő sorrendbe tesszük a sorokat
        // (A második oszlop őrzi az eredeti sorrendet, vagyis a token azonosítóját.)
        float[][] sortedLogits = util.sort(indexedLogits);

        // Hagyjuk meg a legjobb valamennyi (top k) értéket, a többi nem kell
        // (Nem tároltam el újra az index oszlopot is, mert azt úgyis meg lehet találni a sortedLogits táblázatban.)
        float[] filteredLogits = new float[config.topK];
        for (int i = 0; i < config.topK; i++)
        {
            filteredLogits[i] = sortedLogits[i][0];
        }

        // A logit értékek valószínűséggé konvertálása a softmax függvény segítségével
        float[] probabilities = util.softmax(filteredLogits);

        // Súlyozott random választással válasszunk ki egy értéket a megmaradtak közül
        int index = util.weightedRandomPick(probabilities);

        // Olvassuk ki az ehhez tartozó token azonosítót
        int selectedTokenId = (int)sortedLogits[index][1];

        // Kiírjuk az aktuálisan kiválasztott token értékét egyesével
        // Nem tökéletes megoldás, hogy ezt itt írjuk ki, mert néhány betű vagy szó több tokenből áll,
        // de túl lassú a rendszer ahhoz, hogy az egész végét kivárjuk, tehát kiírom inkább őket azonnal.
        PrintStream out = new PrintStream(System.out, true, UTF_8);
        out.print(config.tokenizer.decode(List.of(selectedTokenId)));

        return selectedTokenId;
    }

    /**
     * A dekóderekben áltárolt értékek törlése, hogy egy új menet kezdődhessen
     */
    public void clear()
    {
        for (TransformerDecoder decoder : decoders)
        {
            decoder.clear();
        }
    }
}
