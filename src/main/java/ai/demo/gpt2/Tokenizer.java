package ai.demo.gpt2;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Tokenizer
{
    public static final String ENCODER_FILENAME = "encoder.json";
    public static final String VOCAB_FILENAME = "vocab.bpe";

    private final Map<Integer, Character> charEncoding = new HashMap<>(256); // byte_encoder

    private final Map<Integer, String> tokenEncoding = new HashMap<>(50257); // decoder
    private final Map<String, Integer> tokenDecoding = new HashMap<>(50257); // encoder

    private final Map<Pair, Integer> merges = new HashMap<>(50000); // bpe_ranks

    private final Pattern pattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    public static class Pair
    {
        public final String left;
        public final String right;

        public Pair(String left, String right)
        {
            this.left = left;
            this.right = right;
        }

        @Override
        public boolean equals(Object o)
        {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair pair = (Pair) o;
            return Objects.equals(left, pair.left) && Objects.equals(right, pair.right);
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(left, right);
        }
    }

    public Tokenizer(String path)
    {
        addCharRange(0, 'Ā', 'Ġ');
        addCharRange(33, '!', '~');
        addCharRange(127, 'ġ', 'ł');
        addCharRange(161, '¡', '¬');
        addCharRange(173, 'Ń', 'Ń');
        addCharRange(174, '®', 'ÿ');

        readEncoderFile(path);
        readVocabFile(path);
    }

    private void addCharRange(int pos, char firstChar, char lastChar)
    {
        for (int i = firstChar; i <= lastChar; i++)
        {
            charEncoding.put(pos, (char) i);
            pos++;
        }
    }

    public List<Integer> encode(String text)
    {
        List<Integer> result = new ArrayList<>();

        Matcher matcher = pattern.matcher(text);
        List<String> unicodes = new ArrayList<>();

        while (matcher.find())
        {
            StringBuilder match = new StringBuilder();

            ByteBuffer buffer = StandardCharsets.UTF_8.encode(matcher.group());
            while (buffer.hasRemaining())
            {
                int value = (int)buffer.get();
                if (value < 0) value = value & 0xff;
                match.append(charEncoding.get(value));
            }

            unicodes.add(match.toString());
        }

        for (String word : unicodes)
        {
            for (String token : bpe(word).split(" "))
            {
                Integer value = tokenDecoding.get(token);
                if (value != null)
                {
                    result.add(value);
                }
            }
        }

        return result;
    }

    // TODO: The decoder can't process correctly all special characters
    public String decode(List<Integer> tokens)
    {
        StringBuilder textBuilder = new StringBuilder();
        for (int token : tokens)
        {
            textBuilder.append(tokenEncoding.get(token));
        }
        String text = textBuilder.toString();

        List<String> byteBufferList = new ArrayList<>();
        for (int i = 0; i < text.length(); i++)
        {
            // Special characters, with multiple bytes are failing here (like 'ő')
            // The code of the character is above 255, so it isn't in the charEncoding
            Character chr = charEncoding.get((int)text.charAt(i));
            byteBufferList.add(chr == null ? null : "" + chr);
        }

        byte[] byteBuffer = new byte[byteBufferList.size()];
        for (int i = 0; i < byteBuffer.length; i++)
        {
            String byteString = byteBufferList.get(i);
            if (byteString == null)
            {
                byteString = " ";
            }

            byteBuffer[i] = (byte)byteString.charAt(0);
        }

        return StandardCharsets.UTF_8.decode(ByteBuffer.wrap(byteBuffer)).toString();
    }

    private void readEncoderFile(String path)
    {
        try
        {
            String fileName = path + "/" + ENCODER_FILENAME;

            File file = new File(fileName);
            FileInputStream inputStream = new FileInputStream(file);

            ObjectMapper mapper = new ObjectMapper();

            Map<String, Object> entries = mapper.readValue(inputStream, Map.class);

            for (Map.Entry<String, Object> entry: entries.entrySet())
            {
                Object value = entry.getValue();

                if (value instanceof Integer)
                {
                    tokenEncoding.put((int) entry.getValue(), entry.getKey());
                    tokenDecoding.put(entry.getKey(), (int) entry.getValue());
                }
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    private void readVocabFile(String path)
    {
        try
        {
            String fileName = path + "/" + VOCAB_FILENAME;

            File file = new File(fileName);
            FileInputStream inputStream = new FileInputStream(file);

            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));

            reader.readLine(); // The first line is a comment

            int i = 0;
            while (true)
            {
                String line = reader.readLine();

                if (line == null) break;

                String[] pairs = line.split(" ");
                merges.put(new Pair(pairs[0], pairs[1]), i);

                i++;
            }

            reader.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    private List<Pair> getPairs(List<String> word)
    {
        List<Pair> pairs = new ArrayList<>();

        String prev = word.get(0);

        for (String character : word.subList(1, word.size()))
        {
            pairs.add(new Pair(prev, character));
            prev = character;
        }

        return pairs;
    }

    public String bpe(String token)
    {
        if (token == null || token.length() < 2) return token;

        List<String> word = new ArrayList<>();
        for (char c : token.toCharArray())
        {
            word.add(String.valueOf(c));
        }

        List<Pair> pairs = getPairs(word);

        while (true)
        {
            Pair bigram = findFirstPair(pairs);
            if (bigram == null) break;

            List<String> newWord = new ArrayList<>();

            int i = 0;
            while (i < word.size())
            {
                int j = findFromIndex(word, bigram.left, i);

                if (j != -1)
                {
                    newWord.addAll(word.subList(i, j));
                    i = j;
                }
                else
                {
                    newWord.addAll(word.subList(i, word.size()));
                    break;
                }

                if (word.get(i).equals(bigram.left) && i < word.size() - 1 && word.get(i + 1).equals(bigram.right))
                {
                    newWord.add(bigram.left + bigram.right);
                    i = i + 2;
                }
                else
                {
                    newWord.add(word.get(i));
                    i++;
                }
            }

            word = newWord;

            if (word.size() == 1)
            {
                break;
            }
            else
            {
                pairs = getPairs(word);
            }
        }

        return String.join(" ", word);
    }

    private int findFromIndex(List<String> in, String find, int from)
    {
        for (int i = from; i < in.size(); i++)
        {
            if (in.get(i).equals(find)) return i;
        }

        return -1;
    }

    public Pair findFirstPair(List<Pair> pairs)
    {
        int min = Integer.MAX_VALUE;
        Pair minPair = null;

        for (Pair pair : pairs)
        {
            Integer value = merges.get(pair);

            if (value != null && value.compareTo(min) < 0)
            {
                min = value;
                minPair = pair;
            }
        }

        return minPair;
    }
}
