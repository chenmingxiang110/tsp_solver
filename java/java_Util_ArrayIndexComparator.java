import java.util.Comparator;

public class java_Util_ArrayIndexComparator implements Comparator<Integer>
{
    private final double[] array;
    private final boolean reverse;

    public java_Util_ArrayIndexComparator(double[] array, boolean reverse)
    {
        this.array = array;
        this.reverse = reverse;
    }

    public Integer[] createIndexArray()
    {
        Integer[] indexes = new Integer[array.length];
        for (int i = 0; i < array.length; i++)
        {
            indexes[i] = i; // Autoboxing
        }
        return indexes;
    }

    @Override
    public int compare(Integer index1, Integer index2)
    {
        if (reverse) return array[index1]<(array[index2])?1:array[index1]>(array[index2])?-1:0;
        return array[index1]>(array[index2])?1:array[index1]<(array[index2])?-1:0;
    }
}