import java.io.*;
import java.util.List;
import static java.nio.charset.StandardCharsets.UTF_8;

public class Application
{
    public static final PrintStream OUT = new PrintStream(System.out, true, UTF_8);

    public static void main(String... args) throws Exception
    {
        OUT.println("  _____________________________      ________        ___");
        OUT.println(" /  _____/\\______   \\__    ___/      \\_____  \\    __|  /____   _____   ____");
        OUT.println("/   \\  ___ |     ___/ |    |  ______  /  ____/   / __ |/ __ \\ /     \\ /  _ \\");
        OUT.println("\\    \\_\\  \\|    |     |    | /_____/ /       \\  / /_/ \\  ___/|  Y Y  (  <_> )");
        OUT.println(" \\________/|____|     |____|         \\________\\ \\_____|\\_____>__|_|__/\\____/\n");

        Config config = init(args);

        // Load trained parameter files
        OUT.print("\nLoading trained parameters... ");
        Parameters parameters = new Parameters(config);
        OUT.println("Done.");

        OUT.println("Free memory: " + formatMemorySize(Runtime.getRuntime().freeMemory()));

        Transformer transformer = new Transformer(config, parameters);

        while (true)
        {
            // Read input text
            OUT.print("\nInput text: ");
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String input = reader.readLine();
            if (input.equalsIgnoreCase("q")) break;

            // Split the input text into tokens
            List<Integer> inputTokens = config.tokenizer.encode(input);

            // Use the Transformer
            List<Integer> outputTokens = transformer.processTokens(inputTokens);

            // Convert the output to text and print it
            String response = config.tokenizer.decode(outputTokens);
            OUT.println(/*response*/); // Commented out because the system is slow, and we printed already the text (token by token)

            // Starting a completely new session with every input, because this system isn't for chat
            transformer.clear();
        }
    }

    private static Config init(String[] args)
    {
        // Default values
        ModelType modelType = ModelType.SMALL;
        UtilType utilType = UtilType.ND4J;
        String parametersPath = System.getProperty("user.dir") + "/parameters";
        int maxLength = 25;
        int topK = 40;

        if (args != null)
        {
            // Iterate over the passed parameters and override the default values
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
                    OUT.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                }
            }
        }

        OUT.println("Model type: " + modelType);
        OUT.println("Utility type: " + utilType);
        OUT.println("Parameter path: " + parametersPath);
        OUT.println("Max length: " + maxLength);
        OUT.println("TopK: " + topK);

        // Memory check
        long maxMemory = Runtime.getRuntime().maxMemory();

        if (modelType.minMemory * 1024 * 1024 > maxMemory)
        {
            OUT.println("\nERROR: Not enough memory to load parameters! Minimum memory: " + modelType.minMemory + " MByte.");
            OUT.println("Available memory: " + formatMemorySize(maxMemory));
            OUT.println("You can configure the available memory using the -Xmx and -Xms java flags.");
            OUT.println("(See the batch files.)");

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
            OUT.println("\nWARNING: The provided value can't be converted to integer (" + value + "). Default value will be used.\n");
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
