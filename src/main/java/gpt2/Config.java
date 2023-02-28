package gpt2;

public class Config
{
    public final ModelType modelType;
    public final String parametersPath;
    public final Tokenizer tokenizer;
    public final int maxLength;
    public final int topK;

    public Config(ModelType modelType, String parametersPath, Tokenizer tokenizer, int maxLength,
                  int topK)
    {
        this.modelType = modelType;
        this.parametersPath = parametersPath;
        this.tokenizer = tokenizer;
        this.maxLength = maxLength;
        this.topK = topK;
    }
}
