import java.awt.*;
import java.sql.SQLOutput;
import java.util.*;


public class Main {
    private static int width;
    private static int height;
    private static Scanner sc;
    private static GridCells gridCells;
    private static ViewGrid viewGrid;

    public static void main(String[] args) {

        sc= new Scanner(System.in);
        System.out.println("Please input the width");
        width = sc.nextInt();
        System.out.println("Please input Height");
        height = sc.nextInt();
        System.out.println("Please input the amount of iterations you want");
        int iterations = sc.nextInt();

        gridCells = new GridCells(height, width);
        /*
        First Pattern
        */
/*        gridCells.forceAlive(9,20);
        gridCells.forceAlive(10,20);
        gridCells.forceAlive(9,21);
        gridCells.forceAlive(10,21);*/
        /*
        Second Pattern
        */
        gridCells.forceAlive(9,21);
        gridCells.forceAlive(9,23);
        gridCells.forceAlive(10,20);
        gridCells.forceAlive(11,20);
        gridCells.forceAlive(12,20);
        gridCells.forceAlive(13,20);
        gridCells.forceAlive(13,21);
        gridCells.forceAlive(13,22);
        gridCells.forceAlive(12,23);
        /*
        Third Pattern
         */
/*        gridCells.forceAlive(9,20);
        gridCells.forceAlive(9,21);
        gridCells.forceAlive(10,21);
        gridCells.forceAlive(10,20);

        gridCells.forceAlive(10,16);
        gridCells.forceAlive(11,16);
        gridCells.forceAlive(12,16);
        gridCells.forceAlive(10,15);
        gridCells.forceAlive(11,14);*/



        viewGrid = new ViewGrid(height, height, width);
        for (int i = 0; i < iterations; i++) {
            viewGrid.removeAll();
            viewGrid.addButtons(gridCells.itrGen());
        }
    }
}
