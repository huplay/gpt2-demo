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

        // Tanítás során előállított paraméterek betöltése

        OUT.print("\nBetanított paraméterek betöltése... ");
        Parameters parameters = new Parameters(config);
        OUT.println("Kész.");

        OUT.println("Szabad memória: " + formatMemorySize(Runtime.getRuntime().freeMemory()));

        Transformer transformer = new Transformer(config, parameters);

        while (true)
        {
            // Bemenő szöveg bekérése
            OUT.print("\nInput szöveg: ");
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String input = reader.readLine();
            if (input.equalsIgnoreCase("q")) break;

            // Szöveg tokenekké szabdalása
            List<Integer> inputTokens = config.tokenizer.encode(input);

            // Transformer meghívása
            List<Integer> outputTokens = transformer.processTokens(inputTokens);

            // Eredmény szöveggé alakítása és kiírása
            String response = config.tokenizer.decode(outputTokens);
            OUT.println(/*response*/); // A lassú válaszidő miatt tokenenként kiírtuk már a szöveget, ezért ez a végső kiírás ki lett kommentezve

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
                    OUT.println("\nFigyelem: Értelmezhetetlen paraméter: " + arg + "\n");
                }
            }
        }

        OUT.println("Modell típusa: " + modelType);
        OUT.println("Utility típusa: " + utilType);
        OUT.println("Paraméterek útvonala: " + parametersPath);
        OUT.println("Maximum hosszúság: " + maxLength);
        OUT.println("TopK: " + topK);

        // Memória-méret ellenőrzés
        long maxMemory = Runtime.getRuntime().maxMemory();

        if (modelType.minMemory * 1024 * 1024 > maxMemory)
        {
            OUT.println("\nHIBA: Nincs elég memória a paraméterek betöltéséhez! Minimum memoria: " + modelType.minMemory + " MByte.");
            OUT.println("Szabad memória: " + formatMemorySize(maxMemory));
            OUT.println("Az elérhető memória mérete az -Xmx és -Xms java paraméterekkel állítható.");
            OUT.println("(Lásd a .bat fájlokat!)");

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
            OUT.println("\nFigyelem: A megadott érték nem konvertálható egész számmá (" + value + "). Az alapértelmezett érték lesz használva.\n");
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
