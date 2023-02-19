package ai.demo.gpt2;

import ai.demo.gpt2.util.UtilType;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;

public class Application
{
    public static void main(String... args) throws Exception
    {
        System.out.println("  _____________________________      ________        ___");
        System.out.println(" /  _____/\\______   \\__    ___/      \\_____  \\    __|  /____   _____   ____");
        System.out.println("/   \\  ___ |     ___/ |    |  ______  /  ____/   / __ |/ __ \\ /     \\ /  _ \\");
        System.out.println("\\    \\_\\  \\|    |     |    | /_____/ /       \\  / /_/ \\  ___/|  Y Y  (  <_> )");
        System.out.println(" \\________/|____|     |____|         \\________\\ \\_____|\\_____>__|_|__/\\____/\n");

        Config config = init(args);

        // Tanítás során előállított paraméterek betöltése
        System.out.print("\nLoading trained parameters... ");
        Parameters parameters = new Parameters(config);
        System.out.println("Done.");

        System.out.println("Free memory: " + formatMemorySize(Runtime.getRuntime().freeMemory()));

        Transformer transformer = new Transformer(config, parameters);

        while (true)
        {
            // Bemenő szöveg bekérése
            System.out.print("\nInput text: ");
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String input = reader.readLine();
            if (input.equalsIgnoreCase("q")) break;

            // Szöveg tokenekké szabdalása
            List<Integer> inputTokens = config.tokenizer.encode(input);

            // Transformer meghívása
            List<Integer> outputTokens = transformer.processTokens(inputTokens);

            // Eredmény szöveggé alakítása és kiírása
            String response = config.tokenizer.decode(outputTokens);
            System.out.println(/*response*/); // Commented out because the system is slow, and we printed already the text (token by token)

            // Minden alkalommal egy teljesen új menetet kezdünk, mert ez a rendszer nem csetelésre való
            transformer.clear();
        }
    }

    private static Config init(String[] args)
    {
        // Default értékek
        ModelType modelType = ModelType.SMALL;
        UtilType utilType = UtilType.ND4J;
        String parametersPath = System.getProperty("user.dir") + "/parameters";
        int maxLength = 25;
        int topK = 40;

        if (args != null)
        {
            // Végismegyünk az átadott paramétereken, és felülírjuk a default értékeket
            for (String arg : args)
            {
                String[] parts = arg.split("=");
                if (parts.length == 2)
                {
                    String param = parts[0].toLowerCase();
                    String value = parts[1];

                    switch (param)
                    {
                        case "model" -> modelType = ModelType.find(value);
                        case "util" -> utilType = UtilType.find(value);
                        case "path" -> parametersPath = value;
                        case "maxlength" -> maxLength = readInt(value, maxLength);
                        case "topk" -> topK = readInt(value, topK);
                    }
                }
                else
                {
                    System.out.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                }
            }
        }

        System.out.println("Model type: " + modelType);
        System.out.println("Utility type: " + utilType);
        System.out.println("Parameter path: " + parametersPath);
        System.out.println("Max length: " + maxLength);
        System.out.println("TopK: " + topK);

        // Memória-méret ellenőrzés
        long maxMemory = Runtime.getRuntime().maxMemory();

        if (modelType.minMemory * 1024 * 1024 > maxMemory)
        {
            System.out.println("\nERROR: Not enough memory to load parameters! Minimum memory: " + modelType.minMemory + " MByte.");
            System.out.println("Available memory: " + formatMemorySize(maxMemory));
            System.out.println("You can configure the available memory using the -Xmx and -Xms java flags.");
            System.out.println("(See the batch files.)");

            System.exit(0);
        }

        Tokenizer tokenizer = new Tokenizer(parametersPath);

        return new Config(modelType, utilType, parametersPath, tokenizer, maxLength, topK);
    }

    private static int readInt(String value, int defaultValue)
    {
        int ret = defaultValue;

        try
        {
            ret = Integer.parseInt(value);
        }
        catch (Exception e)
        {
            System.out.println("\nWARNING: The provided value can't be converted to integer (" + value + "). Default value will be used.\n");
        }

        return ret;
    }

    private static String formatMemorySize(long size)
    {
        if (size < 1024) return size + " byte";

        String[] units = new String[] {"kByte", "MByte", "GByte", "PByte", "EByte"};
        int scale = (63 - Long.numberOfLeadingZeros(size)) / 10;
        double quantity = (double) size / (1L << (scale * 10));
        return String.format("%.1f %s", quantity, units[scale - 1]);
    }
}
