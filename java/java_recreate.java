import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

public class java_recreate {

    public static void main(String[] args) throws IOException {
        // String data_root = "tmp_datas/";
        // String state_path = "states.txt";
        // String distMat_path = "distMat.txt";
        String data_root = args[0];
        String state_path = args[1];
        String distMat_path = args[2];
        
        StringBuilder result = new StringBuilder();

        String content = new String(Files.readAllBytes(Paths.get(state_path)), "UTF-8");
        String[] lines = content.split("\n");
        ArrayList<Integer> batch_caps = new ArrayList<>();
        ArrayList<ArrayList<Integer>> batch_ruins = new ArrayList<>();
        ArrayList<ArrayList<int[]>> batch_routes = new ArrayList<>();
        for (String line : lines) {
            String[] line_split = line.split(":");

            ArrayList<Integer> ruins = new ArrayList<>();
            for (String r : line_split[1].split(",")) ruins.add(Integer.parseInt(r));

            ArrayList<int[]> routes = new ArrayList<>();
            for (String route_str: line_split[2].split(";")) {
                String[] route_arr = route_str.split(",");
                int[] route = new int[route_arr.length];
                for (int i = 0 ; i<route_arr.length ; i++) { route[i] = Integer.parseInt(route_arr[i]); }
                routes.add(route);
            }

            batch_caps.add(Integer.parseInt(line_split[0]));
            batch_ruins.add(ruins);
            batch_routes.add(routes);
        }

        ArrayList<double[][]> datas = new ArrayList<>();
        for (int i = 0 ; i<lines.length ; i++) {
            datas.add(Util_SimpleQuestionReader(data_root+"data_"+Integer.toString(i)+".txt"));
        }

        ArrayList<double[][]> distance_matrices = new ArrayList<>();
        content = new String(Files.readAllBytes(Paths.get(distMat_path)), "UTF-8");
        lines = content.split("\n");
        for (String line : lines) {
            distance_matrices.add(Util_SimpleDistMatReader(line));
        }

        for (int i = 0 ; i<datas.size() ; i++) {
            ArrayList<int[]> new_routes = step(datas.get(i),
                    batch_caps.get(i),
                    distance_matrices.get(i),
                    batch_ruins.get(i),
                    batch_routes.get(i));
            double new_distance = getRoutesDistance(new_routes, distance_matrices.get(i));
            result.append(new_distance+":"+routes2str(new_routes)+"\n");
        }
        System.out.println(result.toString());
    }

    public static double[][] Util_SimpleDistMatReader(String DistMat) {
        String[] lines = DistMat.split(";");
        double[][] distance_matrix = new double[lines.length][lines.length];
        for (int i = 0 ; i<lines.length ; i++) {
            String[] dists_str = lines[i].split(",");
            for (int j = 0 ; j<dists_str.length ; j++) {
                distance_matrix[i][j] = Double.parseDouble(dists_str[j]);
            }
        }
        return distance_matrix;
    }

    public static double[][] Util_SimpleQuestionReader(String path) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(path)), "UTF-8");
        String[] lines = content.split("\n");
        double[][] data = new double[lines.length][6];
        for (int i = 0 ; i<lines.length ; i++) {
            String[] d = lines[i].split("\\s+");
            for (int j = 0 ; j<d.length ; j++) {
                data[i][j] = Double.parseDouble(d[j]);
            }
        }
        return data;
    }

    public static String routes2str(ArrayList<int[]> routes) {
        StringBuilder sb = new StringBuilder();
        for (int[] r : routes) {
            for (int i : r) {
                sb.append(i);
                sb.append(",");
            }
            sb.deleteCharAt(sb.length()-1);
            sb.append(";");
        }
        sb.deleteCharAt(sb.length()-1);
        return sb.toString();
    }

    // 0 is not included in the route but will be calculated.
    public static double getRouteDistance(int[] route, double[][] distance_matrix) {
        int last_node = 0;
        double distance = 0.0;
        for (int c : route) {
            distance += distance_matrix[last_node][c];
            last_node = c;
        }
        distance += distance_matrix[last_node][0];
        return distance;
    }

    // 0 is not included in routes but will be calculated.
    public static double getRoutesDistance(ArrayList<int[]> routes, double[][] distance_matrix) {
        double total_distance = 0.0;
        for (int[] r : routes) {
            total_distance += getRouteDistance(r, distance_matrix);
        }
        return total_distance;
    }

    private static ArrayList<int[]> ruin_routeSummary(ArrayList<int[]> lastRoute, ArrayList<Integer> absents) {
        ArrayList<int[]> absentRoute = new ArrayList<int[]>();
        for (int i = 0 ; i<lastRoute.size() ; i++) {
            ArrayList<Integer> r = new ArrayList<>();
            for (int c : lastRoute.get(i)) {
                if (!absents.contains(c)) r.add(c);
            }
            if (r.size()>0) {
                absentRoute.add(r.stream().mapToInt(j->j).toArray());
            }
        }
        return absentRoute;
    }

    private static int[] insertNode(int[] old_r, int pos, int c) {
        int[] new_r = new int[old_r.length+1];
        for (int i = 0 ; i<new_r.length ; i++) {
            if (i<pos) {
                new_r[i] = old_r[i];
            } else if (i>pos) {
                new_r[i] = old_r[i-1];
            } else {
                new_r[i] = c;
            }
        }
        return new_r;
    }

    private static boolean checkValid(double[][] data, double[][] distance_matrix, int[] r, int c) {
        double time_current = 0;
        int curr_node = 0;
        for (int i = 0 ; i<(r.length+1) ; i++) {
            int next_node = i==r.length?0:r[i];
            time_current+=distance_matrix[curr_node][next_node];
            time_current=Math.max(data[next_node][3], time_current);
            if (time_current<=data[next_node][4]) {
                time_current+=data[next_node][5];
            } else {
                return false;
            }
            curr_node = i==r.length?0:r[i];
        }
        return true;
    }

    private static ArrayList<double[]> getValid(double[][] data, double[][] distance_matrix, int[] r, int c) {
        ArrayList<double[]> valids = new ArrayList<>();
        double dist = getRouteDistance(r, distance_matrix);
        double tmp_time = 0;
        int curr_node = 0;
        for (int i = 0 ; i<(r.length+1) ; i++) {
            int next_node = i==r.length?0:r[i];
            tmp_time = Math.max(tmp_time, data[curr_node][3]);
            tmp_time+=data[curr_node][5];
            if (tmp_time+distance_matrix[curr_node][c]>data[c][4]) break;
            int[] new_r = insertNode(r, i, c);
            if (checkValid(data, distance_matrix, new_r, c)) {
                double new_dist = getRouteDistance(new_r, distance_matrix);
                valids.add(new double[]{i, new_dist-dist});
            }
            tmp_time+=distance_matrix[curr_node][next_node];
            curr_node = i==r.length?0:r[i];
        }
        return valids;
    }

    private static ArrayList<int[]> route_add(ArrayList<int[]> absent_route, int c, double[] adding_pos) {
        if (adding_pos[0]==-1) {
            absent_route.add(new int[]{c});
            return absent_route;
        }
        int[] new_r = insertNode(absent_route.get((int)adding_pos[0]), (int)adding_pos[1], c);
        absent_route.set((int)adding_pos[0], new_r);
        return absent_route;
    }

    private static ArrayList<int[]> recreate(double[][] data, double capcity, double[][] distance_matrix,
                                             ArrayList<int[]> absent_route, ArrayList<Integer> absents, int lastLength) {
        Collections.shuffle(absents);
        ArrayList<Integer> newAbsents = new ArrayList<>();
        ArrayList<Integer> toKeep = new ArrayList<>();
        ArrayList<int[]> current_route = absent_route;
        for (int i = 0 ; i<absents.size() ; i++) {
            int c = absents.get(i);
            ArrayList<double[]> probablePlace = new ArrayList<>();
            for (int ir = 0 ; ir<absent_route.size() ; ir++) {
                int[] r = absent_route.get(ir);
                double dmd_sum = 0;
                for (int _tmp_node : r) dmd_sum+=data[_tmp_node][2];
                if ((dmd_sum+data[c][2])>capcity) continue;
                // all possible int values can round-trip to a double safely.
                ArrayList<double[]> valids = getValid(data, distance_matrix, r, c);
                for (double[] v : valids) probablePlace.add(new double[]{ir,v[0],v[1]});
            }
            double[] adding_pos = new double[]{-1,-1,-1};
            if (probablePlace.size()>0) {
                Collections.sort(probablePlace, new Comparator<double[]>() {
                    public int compare(double[] content0, double[] content1) {
                        if (content0[2]<content1[2]) {
                            return -1;
                        } else if (content0[2]>content1[2]) {
                            return 1;
                        }
                        return 0;
                    }
                });
                adding_pos = probablePlace.get(0);
            } else if (lastLength>0 && lastLength<=current_route.size()) {
                toKeep.add(i);
                continue;
            }
            current_route = route_add(current_route, c, adding_pos);
        }
        for (int i : toKeep) newAbsents.add(absents.get(i));
        absents.clear();
        for (int i : newAbsents) absents.add(i);
        return current_route;
    }

    public static ArrayList<int[]> step(double[][] data, double cap, double[][] distance_matrix,
                                        ArrayList<Integer> ruins, ArrayList<int[]> init_route) {
        ArrayList<int[]> absent_route = ruin_routeSummary(init_route, ruins);
        return recreate(data, cap, distance_matrix, absent_route, ruins, 0);
    }

}
