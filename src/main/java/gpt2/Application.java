package gpt2;

import java.io.*;
import java.util.List;

public class Application
{
    public static PrintStream OUT;

    public static void main(String... args) throws Exception
    {
        OUT = new PrintStream(System.out, true, "utf-8");

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

        OUT.println("\nPlease enter a text that the system should continue.");
        OUT.println("(You can leave it empty. To quit: type 'q'.)");

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
            OUT.println(/*response*/); // Commented out because we printed already the text (token by token)

            // Starting a completely new session with every input, because this system isn't for chat
            transformer.clear();
        }

        OUT.println("\nHave a nice day!");
    }

    private static Config init(String[] args)
    {
        // Default values
        ModelType modelType = ModelType.SMALL;
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

                    if (param.equals("model")) modelType = ModelType.find(value);
                    else if (param.equals("path")) parametersPath = value;
                    else if (param.equals("maxlength")) maxLength = readInt(value, maxLength);
                    else if (param.equals("topk")) topK = readInt(value, topK);
                }
                else
                {
                    OUT.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                }
            }
        }

        OUT.println("Model type: " + modelType
                + " - Number of parameters: " + Math.round(modelType.getParameterSize() / 1000000d) + " M");
        OUT.println("Parameter path: " + parametersPath);
        OUT.println("Maximum length of generated text: " + maxLength);
        OUT.println("Output is selected from the best " + topK + " tokens (topK)");

        Tokenizer tokenizer = new Tokenizer(parametersPath);

        return new Config(modelType, parametersPath, tokenizer, maxLength, topK);
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
}
