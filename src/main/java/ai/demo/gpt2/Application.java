package ai.demo.gpt2;

import ai.demo.gpt2.util.UtilType;

import java.io.*;
import java.util.List;

import static java.nio.charset.StandardCharsets.UTF_8;

public class Application
{
    public static void main(String... args) throws Exception
    {
        PrintStream out = new PrintStream(System.out, true, UTF_8);
        out.println("  _____________________________      ________        ___");
        out.println(" /  _____/\\______   \\__    ___/      \\_____  \\    __|  /____   _____   ____");
        out.println("/   \\  ___ |     ___/ |    |  ______  /  ____/   / __ |/ __ \\ /     \\ /  _ \\");
        out.println("\\    \\_\\  \\|    |     |    | /_____/ /       \\  / /_/ \\  ___/|  Y Y  (  <_> )");
        out.println(" \\________/|____|     |____|         \\________\\ \\_____|\\_____>__|_|__/\\____/\n");

        Config config = init(args);

        // Tanítás során előállított paraméterek betöltése

        out.print("\nBetanított paraméterek betöltése... ");
        Parameters parameters = new Parameters(config);
        out.println("Kész.");

        out.println("Szabad memória: " + formatMemorySize(Runtime.getRuntime().freeMemory()));

        Transformer transformer = new Transformer(config, parameters);

        while (true)
        {
            // Bemenő szöveg bekérése
            out.print("\nInput szöveg: ");
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String input = reader.readLine();
            if (input.equalsIgnoreCase("q")) break;

            // Szöveg tokenekké szabdalása
            List<Integer> inputTokens = config.tokenizer.encode(input);

            // Transformer meghívása
            List<Integer> outputTokens = transformer.processTokens(inputTokens);

            // Eredmény szöveggé alakítása és kiírása
            String response = config.tokenizer.decode(outputTokens);
            out.println(/*response*/); // A lassú válaszidő miatt tokenenként kiírtuk már a szöveget, ezért ez a végső kiírás ki lett kommentezve

            // Minden alkalommal egy teljesen új menetet kezdünk, mert ez a rendszer nem csetelésre való
            transformer.clear();
        }
    }

    private static Config init(String[] args)
    {
        // Default értékek
        PrintStream out = new PrintStream(System.out, true, UTF_8);
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
                    out.println("\nFigyelem: Értelmezhetetlen paraméter: " + arg + "\n");
                }
            }
        }

        out.println("Modell típusa: " + modelType);
        out.println("Utility típusa: " + utilType);
        out.println("Paraméterek útvonala: " + parametersPath);
        out.println("Maximum hosszúság: " + maxLength);
        out.println("TopK: " + topK);

        // Memória-méret ellenőrzés
        long maxMemory = Runtime.getRuntime().maxMemory();

        if (modelType.minMemory * 1024 * 1024 > maxMemory)
        {
            out.println("\nHIBA: Nincs elég memória a paraméterek betöltéséhez! Minimum memoria: " + modelType.minMemory + " MByte.");
            out.println("Szabad memória: " + formatMemorySize(maxMemory));
            out.println("Az elérhető memória mérete az -Xmx és -Xms java paraméterekkel állítható.");
            out.println("(Lásd a .bat fájlokat!)");

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
            PrintStream out = new PrintStream(System.out, true, UTF_8);
            out.println("\nFigyelem: A megadott érték nem konvertálható egész számmá (" + value + "). Az alapértelmezett érték lesz használva.\n");
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
