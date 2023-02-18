package ai.demo.gpt2.util;

public enum UtilType
{
    STANDARD(new StandardUtil()),
    ND4J(new Nd4jUtil());

    public final Util util;

    UtilType(Util util)
    {
        this.util = util;
    }

    public static UtilType find(String name)
    {
        UtilType type = ND4J;

        try
        {
            type = UtilType.valueOf(name.toUpperCase());
        }
        catch (IllegalArgumentException e)
        {
            System.out.println("\nWARNING: The selected utility type does not exists (" + name + "), only STANDARD and ND4J are available.\n");
        }

        return type;
    }
}
