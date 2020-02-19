<template>
  <div>
    <datatable v-bind="$data" />
  </div>
</template>

<script>
import Vue from "vue";
import Datatable from "vue2-datatable-component";

Vue.use(Datatable); // done!

export default {
  data: () => ({}),
  watch: {
    query: {
      handler(query) {
        this.data = this.widget.fetch(query.offset, query.limit)        
      },
      deep: true
    }
  }
};
</script>