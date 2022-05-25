public class GridCells {

    private int height;
    private int width;
    private int[][] grid;

    public GridCells(int height, int width){
        /*
        Constructor
         */
        this.height = height;
        this.width = width;
        this.grid = new int[width][height];
    }

    public void forceAlive(int x, int y){
        /*
        Force the cell to be alive
        according to the required patterns
         */
        this.grid[x][y] = 1;
    }

    private int checkBounds(int x, int y) {
        /*
         If a cell exits the 2D
        array then return the count
        as zero for the neighbor
        that exits the grid
         */
        if (x < 0 || x >= width) {
            return 0;
        }

        else if (y < 0 || y >= height) {
            return 0;
        }

        return this.grid[x][y];
    }

    private int getNeighbors(int x, int y){
        /*
        Counts number of alive neighbors 8
        cells surrounding the selected x,y
         */

        int count = 0;

        /*
        Left & Right on same level
         */
        count += checkBounds(x-1,y);
        count += checkBounds(x+1,y);
        /*
        Left, Right, & Center on level below
         */
        count += checkBounds(x-1,y-1);
        count += checkBounds(x,y-1);
        count += checkBounds(x+1,y-1);

        /*
        Left, Right, & Center on level above
         */
        count += checkBounds(x-1,y+1);
        count += checkBounds(x,y+1);
        count += checkBounds(x+1,y+1);


        return count;
    }

    public int[][] itrGen(){
        /*
        Temp grid created to hold the new iteration
        and then pass it to the original grid
        after implementing the conditions of Conway
        */
        int[][] tempGrid = new int[width][height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int neighbors = getNeighbors(x,y);

                /*
                If the cell is alive
                 */
                if(checkBounds(x,y) == 1){
                    /*
                    if the neighbors are less than 2
                    then kill the cell
                     */
                    if (neighbors < 2){
                    tempGrid[x][y] = 0;
                    }
                    /*
                    if the neighbors are equal to 2
                    or 3 then keep alive
                     */
                    else if(neighbors == 2 || neighbors == 3){
                    tempGrid[x][y] = 1;
                    }
                    /*
                    if the neighbors are greater than 3
                    then kill
                     */
                    else if(neighbors > 3){
                        tempGrid[x][y] = 0;
                    }
                    /*
                    If the cell is dead, and it has
                    3 alive neighbors then revive it
                     */
                }  else {
                        if(neighbors == 3){
                        tempGrid[x][y] = 1;
                        }
                    }
            }
        }
        /*
        pass the tempGrid into the main grid
         */
        this.grid = tempGrid;
        return this.grid;
    }
}
