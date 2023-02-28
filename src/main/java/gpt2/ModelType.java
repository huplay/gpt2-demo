package gpt2;

public enum ModelType
{
    // 124M
    SMALL(50257, 1024, 768, 12, 12, 1024),

    // 355M
    MEDIUM(50257, 1024, 1024, 24, 16, 2048),

    // 774M
    LARGE(50257, 1024, 1280, 36, 20, 4096),

    // 1558M
    XL(50257, 1024, 1600, 48, 25, 7168);

    // 175000M - Just for comparison: size of the largest GPT-3 (Not implemented here):
    // GPT_3(50257, 2048, 12288, 96, 96, ???);

    public final int tokenCount;
    public final int contextSize;
    public final int embeddingSize;
    public final int decoderCount;
    public final int headCount;
    public final long minMemory;

    ModelType(int tokenCount, int contextSize, int embeddingSize, int decoderCount, int headCount, long minMemory)
    {
        this.tokenCount = tokenCount;
        this.contextSize = contextSize;
        this.embeddingSize = embeddingSize;
        this.decoderCount = decoderCount;
        this.headCount = headCount;
        this.minMemory = minMemory;
    }

    public static ModelType find(String name)
    {
        ModelType modelType = SMALL;

        try
        {
            modelType = ModelType.valueOf(name.toUpperCase());
        }
        catch (IllegalArgumentException e)
        {
            Application.OUT.println("\nWARNING: The selected model type does not exists (" + name + "), only SMALL/MEDIUM/LARGE and XL are available.\n");
        }

        return modelType;
    }
}
