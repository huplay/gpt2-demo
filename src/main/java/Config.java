public class Config
{
    public final ModelType modelType;
    public final UtilType utilType;
    public final String parametersPath;
    public final Tokenizer tokenizer;
    public final int maxLength;
    public final int topK;

    public Config(ModelType modelType, UtilType utilType, String parametersPath, Tokenizer tokenizer, int maxLength,
                  int topK)
    {
        this.modelType = modelType;
        this.utilType = utilType;
        this.parametersPath = parametersPath;
        this.tokenizer = tokenizer;
        this.maxLength = maxLength;
        this.topK = topK;
    }
}
