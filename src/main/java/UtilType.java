public enum UtilType
{
    STANDARD(new UtilStandard()),
    ND4J(new UtilNd4j());

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
            Application.OUT.println("\nWARNING: The selected utility type does not exists (" + name + "), only STANDARD and ND4J are available.\n");
        }

        return type;
    }
}
