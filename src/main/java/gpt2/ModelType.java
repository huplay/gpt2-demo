package gpt2;

public enum ModelType
{
    // 124M
    SMALL(50257, 1024, 768, 12, 12, false),

    // 355M
    MEDIUM(50257, 1024, 1024, 24, 16, false),

    // 774M
    LARGE(50257, 1024, 1280, 36, 20, false),

    // 1558M
    XL(50257, 1024, 1600, 48, 25, false),

    // For comparison these are the GPT-3 sizes:
    // (It uses a spare transformer, which isn't implemented here, and the parameters are not available)
    GPT3_SMALL(50257, 2048, 768, 12, 12, true),
    GPT3_MEDIUM(50257, 2048, 1024, 24, 16, true),
    GPT3_LARGE(50257, 2048, 1536, 24, 16, true),
    GPT3_XL(50257, 2048, 2048, 24, 24, true),
    GPT3_ADA(50257, 2048, 2560, 32, 32, true),
    GPT3_BABBAGE(50257, 2048, 4096, 32, 32, true),
    GPT3_CURIE(50257, 2048, 5140, 40, 40, true),
    // The version known as the "GPT-3":
    GPT3_DAVINCI(50257, 2048, 12288, 96, 96, true),
    GPT3_DAVINCI_V2(50257, 4000, 12288, 96, 96, true),
    GPT3_DAVINCI_V3(50257, 4000, 12288, 96, 96, true);

    public final int tokenCount;
    public final int contextSize;
    public final int embeddingSize;
    public final int decoderCount;
    public final int headCount;
    public final boolean isSparse;

    ModelType(int tokenCount, int contextSize, int embeddingSize, int decoderCount, int headCount, boolean isSparse)
    {
        this.tokenCount = tokenCount;
        this.contextSize = contextSize;
        this.embeddingSize = embeddingSize;
        this.decoderCount = decoderCount;
        this.headCount = headCount;
        this.isSparse = isSparse;
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
            Application.OUT.println("\nWARNING: The selected model type does not exists (" + name +
                    "), only SMALL/MEDIUM/LARGE and XL are available.\n");
        }

        return modelType;
    }

    public long getParameterSize()
    {
        long wteSize = (long) tokenCount * embeddingSize;
        long wpeSize = (long) contextSize * embeddingSize;
        long finalNormSize = embeddingSize * 2L;

        return wteSize + wpeSize + (getDecoderParameterSize() * decoderCount) + finalNormSize;
    }

    private long getDecoderParameterSize()
    {
        long qkvSize = embeddingSize * embeddingSize * 3L + embeddingSize * 3L;
        long projSize = (long) embeddingSize * embeddingSize + embeddingSize;
        long normSize = embeddingSize * 4L;
        long layer1Size = embeddingSize * embeddingSize * 4L + embeddingSize * 4L;
        long layer2Size = embeddingSize * 4L * embeddingSize + embeddingSize;

        return qkvSize + projSize + normSize + layer1Size + layer2Size;
    }

    public long getMinMemory()
    {
        // Every parameter is stored in 4 bytes (float values)
        long parameterSize = getParameterSize() * 4L;

        // The tokenizer stores the vocabulary 3 times (for encoding/decoding and merges, approx 10 bytes per entry)
        long tokenizerVocabSize = tokenCount * 30L;

        // We store vectors of the embedding size for each token in the context, in every decoder
        // Key and value as well (* 2), every number is stored in 4 bytes (float)
        long kvStoreSize = embeddingSize * contextSize * decoderCount * 8L;

        long otherObjectSize = 500 * 1024 * 1024; // It's a guess

        return parameterSize + tokenizerVocabSize + kvStoreSize + otherObjectSize;
    }
}
